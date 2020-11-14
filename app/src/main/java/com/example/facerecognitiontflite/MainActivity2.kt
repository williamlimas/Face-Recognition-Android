package com.example.facerecognitiontflite

import android.Manifest
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import com.example.facerecognitiontflite.faceantispoof.FaceAntiSpoofing
import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet
import com.example.facerecognitiontflite.mtcnn.MTCNN
import com.google.android.gms.vision.CameraSource.PictureCallback
import id.privy.livenessfirebasesdk.common.*
import id.privy.livenessfirebasesdk.event.LivenessEventProvider
import id.privy.livenessfirebasesdk.vision.VisionDetectionProcessor
import kotlinx.android.synthetic.main.activity_custom_liveness.*
import java.io.IOException
import java.util.*

class MainActivity2 : AppCompatActivity(){

    private val successText = "Silahkan lihat ke kamera lagi untuk mengambil foto"
    private val motionInstructions = arrayOf("Lihat ke kiri", "Lihat ke kanan")
    private val MIN_BASE_DATA = 3
    private val TIMEOUT = 3
    private var is_wfo = false

    internal var graphicOverlay: GraphicOverlay? = null
    internal var preview: CameraSourcePreview? = null

    private var visionDetectionProcessor: VisionDetectionProcessor? = null
    private var cameraSource: CameraSource? = null

    private lateinit var mPref: SharedPreferences
    private lateinit var person: Person

    private var err_verified_counter = 0
    private var success = false
    private var isDebug = false

    private lateinit var faceantispoofing: FaceAntiSpoofing
    private lateinit var mobileFaceNet: MobileFaceNet
    private lateinit var mtcnn: MTCNN

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_custom_liveness)

        initView()
        initTFLiteModel()

        // TODO : Download .json base data

        // TODO : is_wfo assignment with the real condition

        loadBaseData("8495")

        startLiveness()

        LivenessEventProvider.getEventLiveData().observe(this, Observer {
            it?.let {
                when {
                    it.getType() == LivenessEventProvider.LivenessEvent.Type.HeadShake -> {
                        onHeadShakeEvent()
                    }

                    it.getType() == LivenessEventProvider.LivenessEvent.Type.Default -> {
                        onDefaultEvent()
                    }
                }
            }
        })
    }

    private fun initView(){
        preview = findViewById(R.id.cameraPreview)
        graphicOverlay = findViewById(R.id.faceOverlay)

        instructions.text = "Lihat ke kamera dan tempatkan wajah pada overlay"
    }

    private fun restartLiveness(){
        preview?.stop()
        LivenessEventProvider.getEventLiveData().postValue(null)
        startLiveness()
        startCameraSource()
    }

    private fun startLiveness(){
        success = false
        if (PermissionUtil.with(this).isCameraPermissionGranted) {
            createCameraSource()
            startHeadShakeChallenge()
        }
        else {
            PermissionUtil.requestPermission(this, 1, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA)
        }
    }

    private fun startHeadShakeChallenge() {
        visionDetectionProcessor?.setVerificationStep(1)
    }

    private fun onHeadShakeEvent() {
        if (!success) {
            success = true
            motionInstruction.text = successText

            visionDetectionProcessor?.setChallengeDone(true)
        }
    }

    @Suppress("DEPRECATION")
    private fun onDefaultEvent() {
        if (success) {
            Handler().postDelayed({
                cameraSource?.takePicture(null, PictureCallback {

                    val verified = processBitmap(true, BitmapFactory.decodeByteArray(it, 0, it.size))
                    if (verified) {
                        // TODO : Add action when face is Verified
                        restartLiveness()
                    } else {
                        restartLiveness()
                    }
                })
            }, 600)
        }
    }

    override fun onResume() {
        super.onResume()
        startCameraSource()
    }

    override fun onPause() {
        super.onPause()
        preview?.stop()
        LivenessEventProvider.getEventLiveData().postValue(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraSource?.release()
    }

    private fun createCameraSource() {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = CameraSource(this, graphicOverlay)
            cameraSource?.setFacing(CameraSource.CAMERA_FACING_FRONT)
        }

        val motion = VisionDetectionProcessor.Motion.values()[Random().nextInt(
            VisionDetectionProcessor.Motion.values().size)]

        when (motion) {
            VisionDetectionProcessor.Motion.Left -> {
                motionInstruction.text = this.motionInstructions[0]
            }

            VisionDetectionProcessor.Motion.Right -> {
                motionInstruction.text = this.motionInstructions[1]
            }
        }

        visionDetectionProcessor = VisionDetectionProcessor()
        visionDetectionProcessor?.apply {
            isSimpleLiveness(true, this@MainActivity2, motion)
            isDebugMode(isDebug)
        }

        cameraSource?.setMachineLearningFrameProcessor(visionDetectionProcessor)
    }

    private fun startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d("CAMERA SOURCE", "resume: Preview is null")
                }
                preview?.start(cameraSource, graphicOverlay)
            } catch (e: IOException) {
                Log.e("CAMERA SOURCE", "CAMERA SOURCE", e)
                cameraSource?.release()
                cameraSource = null
            }
        }
    }

    fun processBitmap(success: Boolean, bitmap: Bitmap?) : Boolean {
        if (bitmap != null) {
            if (success) {
                // Crop face using MTCNN
                val bitmapCrop = MyUtil.cropFace(mtcnn,(BitmapUtils.processBitmap(bitmap)))
                if (bitmapCrop == null) {
                    Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show()
                    return false
                }
                // Check if the person data is less than the minimum number of base data
                if (person.embeddingSize < MIN_BASE_DATA) {
                    Toast.makeText(this, "Face hasn't been registered offline", Toast.LENGTH_SHORT).show()
                    // TODO : Add online face verification
                    // return true if verified
                    // return false if not verified
                    return false
                }
                // Doing Face Anti Spoofing only for WFH
                if (!is_wfo) {
                    val score: Float = faceantispoofing.antiSpoofing(bitmapCrop)
                    if (score > FaceAntiSpoofing.THRESHOLD) {
                        Toast.makeText(this, "Face is spoof", Toast.LENGTH_SHORT).show()
                        return false
                    }
                }
                // Doing face verification using mobile facenet model
                val verified = verify(bitmapCrop)
                if (verified) {
                    Toast.makeText(this, "Face is verified", Toast.LENGTH_SHORT).show()
                    return true
                } else {
                    err_verified_counter += 1
                    if (err_verified_counter > TIMEOUT) {
                        Toast.makeText(this, "Online face verification", Toast.LENGTH_SHORT).show()
                        // TODO : Add online face verification
                        // return true if verified
                        // return false if not verified
                        return false
                    } else {
                        Toast.makeText(this, "Face is not verified", Toast.LENGTH_SHORT).show()
                        return false
                    }
                }
            }
        }
        return false
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        createCameraSource()
    }

    /**
     * Compare Face between the face in bitmap image and Person data
     * @param person : Person data contains the face-embedding for respective user
     * @param bitmap : Input bitmap image containing face image
     */
    private fun verify(bitmapCrop: Bitmap) : Boolean {
        val distances = FloatArray(person.embeddingSize)
        val embeddings = Array(2) { FloatArray(MobileFaceNet.EMBEDDING_SIZE) }

        // Generate face embedding from given bitmap of cropped face using MobileFaceNet model
        embeddings[0] = mobileFaceNet.generateEmbedding(bitmapCrop)

        // Calculate cosine distance
        for (i in 0 until person.embeddingSize) {
            embeddings[1] = person.getEmbedding(i)
            val distance = mobileFaceNet.cosineDistance(embeddings)
            Log.d("DIST", distance.toString())
            distances[i] = distance
        }

        // Find the minimum cosine distance
        val min_distance = distances[MyUtil.findMinimumDistance(distances)]

        // If distance < Threshold : True
        return min_distance <= MobileFaceNet.THRESHOLD
    }

    /**
     * Initialize TensorFlow Lite model
     * MTCNN : face detector
     * MobileFaceNet : Face Embedding model
     */
    private fun initTFLiteModel(){
        try {
            mtcnn = MTCNN(getAssets())
            mobileFaceNet = MobileFaceNet(getAssets())
            faceantispoofing = FaceAntiSpoofing(getAssets())
        } catch (e : IOException) {
            e.printStackTrace()
        }
    }

    /**
     * Load Base Data according to users' NIK as json filename
     * @param nik
     */
    private fun loadBaseData(nik: String){
        mPref = getPreferences(MODE_PRIVATE)
        if (is_wfo){
            // Adjust the MTCNN threshold
            mtcnn.setThreshold(0.55f);
            // Adjust the Mobile Facenet threshold
            mobileFaceNet.setThreshold(0.4f);
            // Load WFO base data
            person = MyUtil.loadSharedPreference(this, mPref, "Person-WFO", nik + "-wfo.json");
        } else {
            // Adjust the MTCNN threshold
            mtcnn.setThreshold(0.6f);
            // Adjust the Mobile Facenet threshold
            mobileFaceNet.setThreshold(0.3f);// Load WFH base data
            person = MyUtil.loadSharedPreference(this, mPref, "Person-WFH", nik + "-wfh.json");
        }
    }
}
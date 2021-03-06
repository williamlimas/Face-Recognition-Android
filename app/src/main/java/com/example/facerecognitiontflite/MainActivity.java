package com.example.facerecognitiontflite;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.facerecognitiontflite.faceantispoof.FaceAntiSpoofing;
import com.example.facerecognitiontflite.livenessfirebase.LivenessApp;
import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet;
import com.example.facerecognitiontflite.mtcnn.MTCNN;
import java.io.IOException;

import id.privy.livenessfirebasesdk.entity.LivenessItem;
import id.privy.livenessfirebasesdk.listener.PrivyCameraLivenessCallBackListener;

public class MainActivity extends AppCompatActivity {
    private MTCNN mtcnn;
    private MobileFaceNet mobileFaceNet;
    private FaceAntiSpoofing faceantispoofing;

    private Button buttonStart;
    private Button buttonWFO;
    private Button buttonWFH;
    private ImageView imageView;
    private TextView resultTextView;

    private Person person;

    private final int MIN_BASE_DATA = 3;
    private final int TIMEOUT = 3;
    private int err_verified_counter = 0;

    private SharedPreferences mPref;
    private boolean is_wfo = true;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initContent();
        initTFLiteModel();

        // TODO : Download .json base data

        // TODO : is_wfo assignment with the real condition
        // is_wfo = get_status_wfo

        loadBaseData("69001535");

        setButtonColor();

        buttonWFO.setOnClickListener(v -> {
            is_wfo = true;
            setButtonColor();
            loadBaseData("69001535");
        });

        buttonWFH.setOnClickListener(v -> {
            is_wfo = false;
            setButtonColor();
            loadBaseData("69001535");
        });

        // Initialize Liveness Detection
        LivenessApp livenessApp = new LivenessApp.Builder(this)
                .setDebugMode(false) //to enable face landmark detection
                .setMotionInstruction("Lihat ke kiri", "Lihat ke kanan")
                .setSuccessText("Berhasil! Silahkan lihat ke kamera lagi untuk mengambil foto")
                .setInstructions("Lihat ke kamera dan tempatkan wajah pada lingakaran hijau")
                .build();

        buttonStart.setOnClickListener(v -> {
            resultTextView.setText("");
            livenessApp.start(new PrivyCameraLivenessCallBackListener() {
                @Override
                public void success(LivenessItem livenessItem) {
                    Bitmap bitmap = livenessItem.getImageBitmap();
                    imageView.setImageBitmap(bitmap);
                    compareFace(person, bitmap);
                }

                @Override
                public void failed(Throwable t) { }
            });
        });
    }

    private void initContent(){
        buttonStart = findViewById(R.id.btn_start);
        buttonWFH = findViewById(R.id.btn_wfh);
        buttonWFO = findViewById(R.id.btn_wfo);
        imageView = findViewById(R.id.image_view);
        resultTextView = findViewById(R.id.result_text_view);
    }

    /**
     * Compare Face between the face in bitmap image and Person data
     * @param person : Person data contains the face-embedding for respective user
     * @param bitmap : Input bitmap image containing face image
     */
    private void compareFace(Person person, Bitmap bitmap){
        // Doing face detection and face crop
        Bitmap bitmapCrop = MyUtil.cropFace(mtcnn, bitmap);
        if (bitmapCrop == null){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return;
        }

        // Check if the person data is less than the minimum number of base data
        if (person.getEmbeddingSize() < MIN_BASE_DATA){
            Toast.makeText(this, "Face hasn't been registered offline", Toast.LENGTH_SHORT).show();
            // TODO : Add online face verification
            return;
        }

        float[] distances = new float[person.getEmbeddingSize()];
        float[][] embeddings = new float[2][MobileFaceNet.EMBEDDING_SIZE];

        // Doing Face Anti Spoofing only for WFH
        if (!is_wfo) {
            float score = faceantispoofing.antiSpoofing(bitmapCrop);
            if (score > FaceAntiSpoofing.THRESHOLD) {
                Toast.makeText(this, "Face is spoof", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        // Generate face embedding from given bitmap of cropped face using MobileFaceNet model
        embeddings[0] = mobileFaceNet.generateEmbedding(bitmapCrop);

        // Calculate cosine distance
        for (int i = 0; i<person.getEmbeddingSize(); i++){
            embeddings[1] = person.getEmbedding(i);
            distances[i] = mobileFaceNet.cosineDistance(embeddings);
        }

        // Find the minimum cosine distance
        float min_distance = distances[MyUtil.findMinimumDistance(distances)];

        // If distance < Threshold : True
        if (min_distance < MobileFaceNet.THRESHOLD){
            // TODO : Add action when face is Verified
            Toast.makeText(this, "Face is verified", Toast.LENGTH_SHORT).show();
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
            resultTextView.setText("Verified");
        } else {
            err_verified_counter += 1;
            if (err_verified_counter > TIMEOUT) {
                Toast.makeText(this, "Online face verification", Toast.LENGTH_SHORT).show();
                // TODO : Add online face verification
            }
            else {
                Toast.makeText(this, "Face is not verified", Toast.LENGTH_SHORT).show();
            }
            resultTextView.setText("Not Verified");
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
        }
    }

    /**
     * Initialize TensorFlow Lite model
     * MTCNN : face detector
     * MobileFaceNet : Face Embedding model
     */
    private void initTFLiteModel(){
        try {
            mtcnn = new MTCNN(getAssets());
            mobileFaceNet = new MobileFaceNet(getAssets());
            faceantispoofing = new FaceAntiSpoofing(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Load Base Data according to users' NIK as json filename
     * @param nik
     */
    private void loadBaseData(String nik){
        mPref = getPreferences(MODE_PRIVATE);
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

    /**
     * Not necessary, only for debugging
     */
    private void setButtonColor(){
        if (is_wfo){
            buttonWFO.setBackgroundColor(getResources().getColor(R.color.colorAccent));
            buttonWFH.setBackgroundColor((android.R.drawable.btn_default));
        } else {
            buttonWFH.setBackgroundColor(getResources().getColor(R.color.colorAccent));
            buttonWFO.setBackgroundColor((android.R.drawable.btn_default));
        }
    }
}
package com.example.facerecognitiontflite;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet;
import com.example.facerecognitiontflite.mtcnn.Align;
import com.example.facerecognitiontflite.mtcnn.Box;
import com.example.facerecognitiontflite.mtcnn.MTCNN;
import com.google.gson.Gson;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {
    private MTCNN mtcnn;
    private MobileFaceNet mobileFaceNet;

    public static Bitmap bitmap;

    private ImageButton imageButton;
    private ImageView imageView;
    private Button btnCompare;
    private TextView resultTextView;

    private SharedPreferences mPref;
    private Person person_wfh;
    private Person person_wfo;

    private final int TIMEOUT = 3;
    private int err_count = 1;

    // TODO : is_wfo assignment with the real condition
    private boolean is_wfo = false;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mPref = getPreferences(MODE_PRIVATE);

        person_wfo = loadSharedPreference("Person-WFO", "1012-wfo.json");
        person_wfh = loadSharedPreference("Person-WFH", "1012-wfh.json");

        initContent();
        initTFLiteModel();
        initCamera();

        btnCompare.setOnClickListener(v -> {
            if (is_wfo) compareFace(person_wfo, bitmap);
            else compareFace(person_wfh, bitmap);
        });
    }

    private void initContent(){
        imageButton = findViewById(R.id.image_button);
        imageView = findViewById(R.id.image_view_crop);
        btnCompare = findViewById(R.id.btn_compare_);
        resultTextView = findViewById(R.id.result_text_view_);
    }

    @SuppressLint("StaticFieldLeak")
    public static ImageButton currentBtn;
    private void initCamera() {
        View.OnClickListener listener = v -> {
            currentBtn = (ImageButton) v;
            startActivity(new Intent(MainActivity.this, CameraActivity.class));
        };
        imageButton.setOnClickListener(listener);
    }

    /**
     * Compare Face between the face in bitmap image and Person data
     * @param person : Person data contains the face-embedding for respective user
     * @param bitmap : Input bitmap image containing face image
     */
    private void compareFace(Person person, Bitmap bitmap){
        float[] distances = new float[person.getEmbeddingSize()];
        float[][] embeddings = new float[2][MobileFaceNet.EMBEDDING_SIZE];

        // Check if error count is more than timeout
        if (err_count > TIMEOUT) {
            Toast.makeText(this, "Online face verification", Toast.LENGTH_SHORT).show();
            // TODO : Add online face verification
        }

        // Check if the person data is not available
        if (person.getEmbeddingSize() == 0){
            Toast.makeText(this, "Face hasn't been registered offline", Toast.LENGTH_SHORT).show();
            // TODO : Add online face verification
            return;
        }

        // Doing face detection and face crop
        Bitmap bitmapCrop = cropFace(bitmap);
        if (bitmapCrop == null){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return;
        }

        // Generate face embedding from given bitmap of cropped face using MobileFaceNet model
        embeddings[0] = mobileFaceNet.generateEmbedding(bitmapCrop);

        // Calculate cosine distance
        for (int i = 0; i<person.getEmbeddingSize(); i++){
            embeddings[1] = person.getEmbedding(i);
            distances[i] = mobileFaceNet.cosineDistance(embeddings);
        }

        // Find the minimum cosine distance
        float min_distance = distances[findMinimumDistance(distances)];

        if (min_distance < MobileFaceNet.THRESHOLD){
            // TODO : Add action when Face is Verified
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
        } else {
            Toast.makeText(this, "Face is not verified", Toast.LENGTH_SHORT).show();
            err_count += 1;
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
        }
        String text = "Cosine Distance : ";
        text = text + min_distance;
        resultTextView.setText(text);
    }

    /**
     * Find Minimum Distance
     * @param data : Float array
     * @return index of the minimum float number
     */
    private int findMinimumDistance(float[] data){
        int idx_min = 0;
        for (int i=1; i<data.length; i++) {
            if (data[i] < data[idx_min]) {
                idx_min = i;
            }
        }
        return idx_min;
    }

    /**
     * Crop Face
     * @param bitmap : Input bitmap image
     * @return Bitmap of cropped faces
     */
    private Bitmap cropFace(Bitmap bitmap){
        if (bitmap == null){
            Toast.makeText(this, "Please take a photo", Toast.LENGTH_SHORT).show();
            return null;
        }

        // Face Alignment
        Bitmap bitmapTemp = bitmap.copy(bitmap.getConfig(), false);
        Vector<Box> boxes = mtcnn.detectFaces(bitmapTemp, bitmapTemp.getWidth() / 5);
        if (boxes.size() == 0){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return null;
        }
        Point[] landmark = boxes.get(MyUtil.findLargestFace(boxes)).landmark;
        bitmapTemp = Align.face_align(bitmapTemp, landmark);

        // Face Detection
        boxes = mtcnn.detectFaces(bitmapTemp, bitmapTemp.getWidth() / 5);
        if (boxes.size() == 0){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return null;
        }
        Box box = boxes.get(MyUtil.findLargestFace(boxes));
        box.toSquareShape();
        box.limitSquare(bitmapTemp.getWidth(), bitmapTemp.getHeight());

        Rect rect = box.transform2Rect();
        Bitmap bitmapCrop = MyUtil.crop(bitmapTemp, rect);
        imageView.setImageBitmap(bitmapCrop);
        return bitmapCrop;
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
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Load Shared Preference
     * @param key : whether the user is WFO or WFH (if WFO then use Person-WFO, else use Person-WFH)
     * @param filename : .json file that contains the face-embedding for respective user
     * @return Person data contains the face-embedding for respective user
     * (if there's no base data then the return will be null)
     */
    private Person loadSharedPreference(String key, String filename){
        // Load the data if already saved in SharedPreference
        Gson gson = new Gson();
        String json = mPref.getString(key, "");
        Person person = gson.fromJson(json, Person.class);

        // Load the data from .json file if the person data is null then save it in Shared Preference
        if (person == null) {
            person = parseJSON(filename);
            saveSharedPreference(person, key);
        }
        return person;
    }

    /**
     * Save Shared Preference
     * @param person : Person data contains the face-embedding for respective user
     * @param key : whether the user is WFO or WFH (if WFO then use Person-WFO, else use Person-WFH)
     */
    private void saveSharedPreference(Person person, String key){
        SharedPreferences.Editor prefsEditor = mPref.edit();
        Gson gson = new Gson();
        String json = gson.toJson(person);
        prefsEditor.putString(key, json);
        prefsEditor.apply();
    }

    /**
     * Parse JSON file
     * @param filename : .json file that contains the face-embedding for respective user
     * @return Person data contains the face-embedding for respective user
     */
    private Person parseJSON(String filename){
        try {
            Person person = new Person();
            JSONObject obj = new JSONObject(loadJSONFromAsset(filename));
            JSONArray str_embeddings = obj.getJSONArray("embedding");

            for (int i=0; i<str_embeddings.length(); i++){
                float[] embedding = new float[MobileFaceNet.EMBEDDING_SIZE];
                int embedding_counter = 0;
                String str_embedding = str_embeddings.get(i).toString().substring(1);

                int j = 0;
                while(j<str_embedding.length()){
                    StringBuilder num = new StringBuilder();
                    while (str_embedding.charAt(j) != '|'){
                        num.append(str_embedding.charAt(j));
                        j += 1;
                        if (j >= str_embedding.length()) break;
                    }
                    j += 1;
                    embedding[embedding_counter] = (Float.parseFloat(num.toString()));
                    embedding_counter += 1;
                }
                person.addEmbedding(embedding);
            }
            return person;
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Load JSON from Assets folder
     * @param filename : .json file that contains the face-embedding for respective user
     * @return String of JSON
     */
    private String loadJSONFromAsset(String filename){
        String json = null;
        try {
            InputStream is = getAssets().open(filename);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return json;
    }
}

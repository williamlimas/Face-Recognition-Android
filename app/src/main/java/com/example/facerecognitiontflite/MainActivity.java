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
            if (is_wfo) compareFace(person_wfo);
            else compareFace(person_wfh);
        });
    }

    private void compareFace(Person person){
        float[] distances = new float[person.getEmbeddingSize()];
        float[][] embeddings = new float[2][MobileFaceNet.EMBEDDING_SIZE];

        if (person.getEmbeddingSize() == 0){
            Toast.makeText(this, "Face hasn't been registered offline", Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap bitmapCrop = cropFace(bitmap);
        if (bitmapCrop == null){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return;
        }

        embeddings[0] = mobileFaceNet.generateEmbedding(bitmapCrop);
        for (int i = 0; i<person.getEmbeddingSize(); i++){
            embeddings[1] = person.getEmbedding(i);
            distances[i] = mobileFaceNet.cosineDistance(embeddings);
        }

        float min_distance = distances[findMinimumDistance(distances)];

        if (min_distance < MobileFaceNet.THRESHOLD){
            // Face Verified
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
        } else {
            // Face Not Verified
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
        }
        String text = "Cosine Distance : ";
        text = text + min_distance;
        resultTextView.setText(text);
    }

    private int findMinimumDistance(float[] data){
        int idx_min = 0;
        for (int i=1; i<data.length; i++) {
            if (data[i] < data[idx_min]) {
                idx_min = i;
            }
        }
        return idx_min;
    }

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

    private void initContent(){
        imageButton = findViewById(R.id.image_button);
        imageView = findViewById(R.id.image_view_crop);
        btnCompare = findViewById(R.id.btn_compare_);
        resultTextView = findViewById(R.id.result_text_view_);
    }

    private void initTFLiteModel(){
        try {
            mtcnn = new MTCNN(getAssets());
            mobileFaceNet = new MobileFaceNet(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }
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

    private Person loadSharedPreference(String key, String filename){
        Gson gson = new Gson();
        String json = mPref.getString(key, "");
        Person person = gson.fromJson(json, Person.class);
        if (person == null)
            person = parse_json(key, filename);
        return person;
    }

    private void saveSharedPreference(Person person, String key){
        SharedPreferences.Editor prefsEditor = mPref.edit();
        Gson gson = new Gson();
        String json = gson.toJson(person);
        prefsEditor.putString(key, json);
        prefsEditor.apply();
    }

    private Person parse_json(String key, String filename){
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
                return person;
            }
            saveSharedPreference(person, key);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return null;
    }

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

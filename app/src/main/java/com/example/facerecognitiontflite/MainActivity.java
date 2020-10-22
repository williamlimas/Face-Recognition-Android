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

import java.io.IOException;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {
    private MTCNN mtcnn;
    private MobileFaceNet mobileFaceNet;

    public static Bitmap bitmap;

    private ImageButton imageButton;
    private ImageView imageView;
    private Button btnCompare;
    private Button btnRegister;
    private TextView resultTextView;

    private SharedPreferences mPref;
    private Person person;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        mPref = getPreferences(MODE_PRIVATE);

        loadJSON();
        initContent();
        initTFLiteModel();
        initCamera();

        btnCompare.setOnClickListener(v -> compareFace());

        btnRegister.setOnClickListener(v -> saveEmbedding());
    }

    private void compareFace(){
        float[] distances = new float[person.getIdx_read()];
        float[][] embeddings = new float[2][MobileFaceNet.EMBEDDING_SIZE];

        if (person.getIdx_read() == 0){
            Toast.makeText(this, "Base data not found", Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap bitmapCrop = cropFace();
        if (bitmapCrop == null){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return;
        }

        embeddings[0] = mobileFaceNet.generateEmbedding(bitmapCrop);
        for (int i = 0; i<person.getIdx_read(); i++){
            embeddings[1] = person.getEmbedding(i);
            distances[i] = mobileFaceNet.cosineDistance(embeddings);
        }

        float min_distance = distances[findMinimumDistance(distances)];

        if (min_distance < MobileFaceNet.THRESHOLD){
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
        } else {
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

    private Bitmap cropFace(){
        if (bitmap == null){
            Toast.makeText(this, "Please take a picture", Toast.LENGTH_SHORT).show();
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
        btnRegister = findViewById(R.id.btn_register_);
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

    private void saveEmbedding(){
        Bitmap bitmapCrop = cropFace();
        if (bitmapCrop == null){
            Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
            return;
        }
        float[] embedding = mobileFaceNet.generateEmbedding(bitmapCrop);
        person.addEmbedding(embedding);
        saveJSON();
    }

    private void loadJSON(){
        Gson gson = new Gson();
        String json = mPref.getString("Person", "");
        person = gson.fromJson(json, Person.class);
        if (person == null)
            person = new Person();
    }

    private void saveJSON(){
        SharedPreferences.Editor prefsEditor = mPref.edit();
        Gson gson = new Gson();
        String json = gson.toJson(person);
        prefsEditor.putString("Person", json);
        prefsEditor.apply();
    }
}

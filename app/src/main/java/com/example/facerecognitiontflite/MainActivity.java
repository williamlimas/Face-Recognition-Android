package com.example.facerecognitiontflite;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet;
import com.example.facerecognitiontflite.mtcnn.Align;
import com.example.facerecognitiontflite.mtcnn.Box;
import com.example.facerecognitiontflite.mtcnn.MTCNN;

import java.io.IOException;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {
    private MTCNN mtcnn;
    private MobileFaceNet mobileFaceNet;

    public static Bitmap bitmap1;
    public static Bitmap bitmap2;
    private Bitmap bitmapCrop1;
    private Bitmap bitmapCrop2;

    private ImageButton imageButton1;
    private ImageButton imageButton2;
    private ImageView imageView1;
    private ImageView imageView2;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initContent();
        initTFLiteModel();
        initCamera();

        Button btnCrop = findViewById(R.id.btn_crop);
        Button btnCompare = findViewById(R.id.btn_compare);

        btnCrop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cropFace();
            }
        });

        btnCompare.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                compareFace();
            }
        });
    }

    private void cropFace(){
        if (bitmap1 == null || bitmap2 == null){
            Toast.makeText(this, "There's empty image", Toast.LENGTH_SHORT).show();
            return;
        }
        Bitmap bitmapTemp1 = bitmap1.copy(bitmap1.getConfig(), false);
        Bitmap bitmapTemp2 = bitmap2.copy(bitmap2.getConfig(), false);

        Point[] landmark1 = getFaceLandmarkMTCNN(bitmapTemp1);
        Point[] landmark2 = getFaceLandmarkMTCNN(bitmapTemp2);

        if (landmark1 == null){
            Toast.makeText(this, "No faces detected in image 1", Toast.LENGTH_SHORT).show();
            return;
        }
        if (landmark2 == null){
            Toast.makeText(this, "No faces detected in image 2", Toast.LENGTH_SHORT).show();
            return;
        }

        bitmapTemp1 = Align.face_align(bitmapTemp1, landmark1);
        bitmapTemp2 = Align.face_align(bitmapTemp2, landmark2);

        Vector<Box> boxes1 = faceDetectMTCNN(bitmapTemp1);
        Vector<Box> boxes2 = faceDetectMTCNN(bitmapTemp2);

        Box box1 = boxes1.get(0);
        Box box2 = boxes2.get(0);

        box1.toSquareShape();
        box2.toSquareShape();

        box1.limitSquare(bitmapTemp1.getWidth(), bitmapTemp1.getHeight());
        box2.limitSquare(bitmapTemp2.getWidth(), bitmapTemp2.getHeight());

        Rect rect1 = box1.transform2Rect();
        Rect rect2 = box2.transform2Rect();

        bitmapCrop1 = MyUtil.crop(bitmapTemp1, rect1);
        bitmapCrop2 = MyUtil.crop(bitmapTemp2, rect2);

        imageView1.setImageBitmap(bitmapCrop1);
        imageView2.setImageBitmap(bitmapCrop2);
    }

    private void compareFace(){
        if (bitmapCrop1 == null || bitmapCrop2 == null){
            Toast.makeText(this, "There's empty image", Toast.LENGTH_SHORT).show();
            return;
        }
        float similarity = mobileFaceNet.compare(bitmapCrop1, bitmapCrop2);
        if (similarity < MobileFaceNet.THRESHOLD){
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
        } else {
            resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
        }
        String text = "Cosine Distance : ";
        text = text + similarity;
        resultTextView.setText(text);
    }

    private Point[] getFaceLandmarkMTCNN(Bitmap bitmap){
        Vector<Box> boxes = faceDetectMTCNN(bitmap);
        if (boxes.size() > 0){
            return boxes.get(0).landmark;
        } else {
            return null;
        }
    }

    private Vector<Box> faceDetectMTCNN(Bitmap bitmap){
        long timeStart = System.currentTimeMillis();
        Vector<Box> boxes = mtcnn.detectFaces(bitmap, bitmap.getWidth() / 5);
        long timeEnd = System.currentTimeMillis();
        Log.d("FACE DETECT", "Elapsed time :" + (timeEnd - timeStart));
        return boxes;
    }

    private void initContent(){
        imageButton1 = findViewById(R.id.image_button1);
        imageButton2 = findViewById(R.id.image_button2);
        imageView1 = findViewById(R.id.image_view_crop1);
        imageView2 = findViewById(R.id.image_view_crop2);
        resultTextView = findViewById(R.id.result_text_view);
    }

    @SuppressLint("StaticFieldLeak")
    public static ImageButton currentBtn;
    private void initCamera() {
        View.OnClickListener listener = v -> {
            currentBtn = (ImageButton) v;
            startActivity(new Intent(MainActivity.this, CameraActivity.class));
        };
        imageButton1.setOnClickListener(listener);
        imageButton2.setOnClickListener(listener);
    }

    private void initTFLiteModel(){
        try {
            mtcnn = new MTCNN(getAssets());
            mobileFaceNet = new MobileFaceNet(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

package com.example.facerecognitiontflite;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.util.Log;
import android.widget.Toast;

import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet;
import com.example.facerecognitiontflite.mtcnn.Align;
import com.example.facerecognitiontflite.mtcnn.Box;
import com.example.facerecognitiontflite.mtcnn.MTCNN;
import com.google.gson.Gson;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Vector;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class MyUtil {

    /**
     * Read pictures from assets
     * @param context
     * @param filename
     * @return
     */
    public static Bitmap readFromAssets(Context context, String filename){
        Bitmap bitmap;
        AssetManager asm = context.getAssets();
        try {
            InputStream is = asm.open(filename);
            bitmap = BitmapFactory.decodeStream(is);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        return bitmap;
    }

    /**
     * Add margin to rect
     * @param bitmap
     * @param rect
     * @param marginX
     * @param marginY
     */
    public static void rectExtend(Bitmap bitmap, Rect rect, int marginX, int marginY) {
        rect.left = max(0, rect.left - marginX / 2);
        rect.right = min(bitmap.getWidth() - 1, rect.right + marginX / 2);
        rect.top = max(0, rect.top - marginY / 2);
        rect.bottom = min(bitmap.getHeight() - 1, rect.bottom + marginY / 2);
    }

    /**
     * Add margin to rect
     * Use the same length, increase the width to the same length
     * @param bitmap
     * @param rect
     */
    public static void rectExtend(Bitmap bitmap, Rect rect) {
        int width = rect.right - rect.left;
        int height = rect.bottom - rect.top;
        int margin = (height - width) / 2;
        rect.left = max(0, rect.left - margin);
        rect.right = min(bitmap.getWidth() - 1, rect.right + margin);
    }

    /**
     * Load TFlite model file
     * @param assetManager
     * @param modelPath
     * @return
     * @throws IOException
     */
    public static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Normalize the picture to [-1, 1]
     * @param bitmap
     * @return
     */
    public static float[][][] normalizeImage(Bitmap bitmap) {
        int h = bitmap.getHeight();
        int w = bitmap.getWidth();
        float[][][] floatValues = new float[h][w][3];

        float imageMean = 127.5f;
        float imageStd = 128;

        int[] pixels = new int[h * w];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, w, h);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                final int val = pixels[i * w + j];
                float r = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                float g = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                float b = ((val & 0xFF) - imageMean) / imageStd;
                float[] arr = {r, g, b};
                floatValues[i][j] = arr;
            }
        }
        return floatValues;
    }

    /**
     * Zoom picture
     * @param bitmap
     * @param scale
     * @return
     */
    public static Bitmap bitmapResize(Bitmap bitmap, float scale) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        return Bitmap.createBitmap(
                bitmap, 0, 0, width, height, matrix, true);
    }

    /**
     * Image transpose
     * @param in
     * @return
     */
    public static float[][][] transposeImage(float[][][] in) {
        int h = in.length;
        int w = in[0].length;
        int channel = in[0][0].length;
        float[][][] out = new float[w][h][channel];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                out[j][i] = in[i][j] ;
            }
        }
        return out;
    }

    /**
     * 4-dimensional image batch matrix width and height transposition
     * @param in
     * @return
     */
    public static float[][][][] transposeBatch(float[][][][] in) {
        int batch = in.length;
        int h = in[0].length;
        int w = in[0][0].length;
        int channel = in[0][0][0].length;
        float[][][][] out = new float[batch][w][h][channel];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    out[i][k][j] = in[i][j][k] ;
                }
            }
        }
        return out;
    }

    /**
     * Intercept the specified rectangular box in the box (cross-border processing),
     * and resize to size*size, then store the returned data.
     * @param bitmap
     * @param box
     * @param size
     * return
     */
    public static float[][][] cropAndResize(Bitmap bitmap, Box box, int size) {
        // crop and resize
        Matrix matrix = new Matrix();
        float scaleW = 1.0f * size / box.width();
        float scaleH = 1.0f * size / box.height();
        matrix.postScale(scaleW, scaleH);
        Rect rect = box.transform2Rect();
        Bitmap croped = Bitmap.createBitmap(
                bitmap, rect.left, rect.top, box.width(), box.height(), matrix, true);

        return normalizeImage(croped);
    }

    /**
     * Cut out the face according to the size of rect
     * @param bitmap
     * @param rect
     * @return
     */
    public static Bitmap crop(Bitmap bitmap, Rect rect) {
        return Bitmap.createBitmap(bitmap, Math.max(0, rect.left), Math.max(0, rect.top),
                Math.abs(rect.right - rect.left), Math.abs(rect.bottom - rect.top));
    }

    /**
     * L2-norm normalization
     * @param embeddings
     * @param epsilon
     * @return
     */
    public static void l2Normalize(float[][] embeddings, double epsilon) {
        for (int i = 0; i < embeddings.length; i++) {
            float squareSum = 0;
            for (int j = 0; j < embeddings[i].length; j++) {
                squareSum += Math.pow(embeddings[i][j], 2);
            }
            float xInvNorm = (float) Math.sqrt(Math.max(squareSum, epsilon));
            for (int j = 0; j < embeddings[i].length; j++) {
                embeddings[i][j] = embeddings[i][j] / xInvNorm;
            }
        }
    }

    /**
     * Picture to grayscale
     * @param bitmap
     * @return 
     */
    public static int[][] convertGreyImg(Bitmap bitmap) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();

        int[] pixels = new int[h * w];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);

        int[][] result = new int[h][w];
        int alpha = 0xFF << 24;
        for(int i = 0; i < h; i++)	{
            for(int j = 0; j < w; j++) {
                int val = pixels[w * i + j];

                int red = ((val >> 16) & 0xFF);
                int green = ((val >> 8) & 0xFF);
                int blue = (val & 0xFF);

                int grey = (int)((float) red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
                grey = alpha | (grey << 16) | (grey << 8) | grey;
                result[i][j] = grey;
            }
        }
        return result;
    }

    /**
     * Crop Face
     * @param bitmap : Input bitmap image
     * @param mtcnn : MTCNN Face Detector
     * @return Bitmap of cropped faces
     */
    public static Bitmap cropFace(MTCNN mtcnn, Bitmap bitmap){
        if (bitmap == null){
            return null;
        }

        // Face Alignment
        Bitmap bitmapTemp = bitmap.copy(bitmap.getConfig(), false);
        Vector<Box> boxes = mtcnn.detectFaces(bitmapTemp, bitmapTemp.getWidth() / 5);
        if (boxes.size() == 0){
            return null;
        }
        Point[] landmark = boxes.get(MyUtil.findLargestFace(boxes)).landmark;
        bitmapTemp = Align.face_align(bitmapTemp, landmark);

        // Face Detection
        boxes = mtcnn.detectFaces(bitmapTemp, bitmapTemp.getWidth() / 5);
        if (boxes.size() == 0){
            return null;
        }
        Box box = boxes.get(MyUtil.findLargestFace(boxes));
        box.toSquareShape();
        box.limitSquare(bitmapTemp.getWidth(), bitmapTemp.getHeight());

        Rect rect = box.transform2Rect();
        return MyUtil.crop(bitmapTemp, rect);
    }

    /**
     * Get Largest Face (if there're two or more faces)
     * @param boxes : Vector of Box that represent the detected faces in Box data type
     * @return index of largest face
     */
    public static int findLargestFace(Vector<Box> boxes){
        int idx = 0;
        if (boxes.size() > 1) {
            for (int i = 1; i < boxes.size(); i++) {
                if (boxes.get(i).width()*boxes.get(i).height() >
                        boxes.get(idx).width()*boxes.get(idx).height())
                    idx = i;
            }
        }
        return idx;
    }

    /**
     * Find Minimum Distance
     * @param data : Float array
     * @return index of the minimum float number
     */
    public static int findMinimumDistance(float[] data){
        int idx_min = 0;
        for (int i=1; i<data.length; i++) {
            if (data[i] < data[idx_min]) {
                idx_min = i;
            }
        }
        return idx_min;
    }

    /**
     * Load Shared Preference
     * @param key : whether the user is WFO or WFH (if WFO then use Person-WFO, else use Person-WFH)
     * @param filename : .json file that contains the face-embedding for respective user
     * @return Person data contains the face-embedding for respective user
     * (if there's no base data then the return will be null)
     */
    public static Person loadSharedPreference(Context context, SharedPreferences mPref, String key, String filename){
        // Load the data if already saved in SharedPreference
        Gson gson = new Gson();
        String json = mPref.getString(key, "");
        Person person = gson.fromJson(json, Person.class);

        // Load the data from .json file if the person data is null then save it in Shared Preference
        if (person == null && !filename.equals("")) {
            person = parseJSON(context, filename);
            saveSharedPreference(mPref, person, key);
        }
        return person;
    }

    /**
     * Save Shared Preference
     * @param person : Person data contains the face-embedding for respective user
     * @param key : whether the user is WFO or WFH (if WFO then use Person-WFO, else use Person-WFH)
     */
    public static void saveSharedPreference(SharedPreferences mPref, Person person, String key){
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
    private static Person parseJSON(Context context, String filename){
        try {
            Person person = new Person();
            JSONObject obj = new JSONObject(loadJSONFromAsset(context, filename));
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
                Log.d("EMB", Arrays.toString(embedding));
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
    private static String loadJSONFromAsset(Context context, String filename){
        String json = null;
        try {
            InputStream is = context.getAssets().open(filename);
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

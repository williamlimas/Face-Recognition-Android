package com.example.facerecognitiontflite.mobilefacenet;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import com.example.facerecognitiontflite.MyUtil;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;

/**
 * MobileFaceNet : Face Verification
 * malikanhar.maulana@wingscorp.com
 */
public class MobileFaceNet {
    public static final String MODEL_FILE = "MobileFaceNet_SE.tflite";

    /**
     * INPUT_IMAGE_SIZE
     * - MobileFaceNet, MobiFace, MobileFaceNet_SE, EfficientNet : 112
     * - FaceNet : 160
     */
    public static final int INPUT_IMAGE_SIZE = 112;

    /**
     * EMBEDDING SIZE
     * - MobileFaceNet 192
     * - EfficientNet, MobileFaceNet_SE : 256
     * - MobiFace, FaceNet : 512
     */
    public static final int EMBEDDING_SIZE = 256;

    public static final float THRESHOLD = 0.4f;

    private Interpreter interpreter;

    public MobileFaceNet(AssetManager assetManager) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(MyUtil.loadModelFile(assetManager, MODEL_FILE), options);
    }

    /**
     * Comparing two bitmap images
     * @param bitmap1 : first bitmap image to be compared
     * @param bitmap2 : second bitmap image to be compared
     * @return Embedding distance
     */
    public float compare(Bitmap bitmap1, Bitmap bitmap2){
        Bitmap bitmapResize1 = Bitmap.createScaledBitmap(bitmap1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE,true);
        Bitmap bitmapResize2 = Bitmap.createScaledBitmap(bitmap2, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE,true);

        float[][][][] datasets = getTwoImageDatasets(bitmapResize1, bitmapResize2);
        float[][] embeddings = new float[2][EMBEDDING_SIZE];

        long timeStart = System.currentTimeMillis();
        interpreter.run(datasets, embeddings);
        long timeEnd = System.currentTimeMillis();
        Log.d("FACE VERIF", "Elapsed time : " + (timeEnd - timeStart));

//        MyUtil.l2Normalize(embeddings, 1e-10);
        return cosineDistance(embeddings);
    }

    public float[] generateEmbedding(Bitmap bitmap){
        Bitmap bitmapResize = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE,true);

        int[] ddims = {1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3};
        float[][][][] datasets = new float[ddims[0]][ddims[1]][ddims[2]][ddims[3]];

        datasets[0] = MyUtil.normalizeImage(bitmapResize);

        float[][] embeddings = new float[1][EMBEDDING_SIZE];
        interpreter.run(datasets, embeddings);
        return embeddings[0];
    }

    /**
     * Calculate L2 distance between two embeddings
     * @param embeddings : face embeddings
     * @return Embedding distance
     */
    public float cosineDistance(float[][] embeddings){
        float[] embeddings1 = embeddings[0];
        float[] embeddings2 = embeddings[1];
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        for (int i=0; i<embeddings1.length; i++){
            dotProduct = dotProduct + (embeddings1[i] * embeddings2[i]);
            normA = (float)(normA + (Math.pow(embeddings1[i], 2.0)));
            normB = (float)(normB + (Math.pow(embeddings1[i], 2.0)));
        }
        float distance = (float)(1-(dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))));
        return Math.min(1, Math.max(0, distance));
    }

    /**
     * Create datasets containing two bitmap images
     * @param bitmap1 : first image bitmap
     * @param bitmap2 : second image bitmap
     * @return float array containing two bitmap images
     */
    private float[][][][] getTwoImageDatasets(Bitmap bitmap1, Bitmap bitmap2) {
        Bitmap[] bitmaps = {bitmap1, bitmap2};

        int[] ddims = {bitmaps.length, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3};
        float[][][][] datasets = new float[ddims[0]][ddims[1]][ddims[2]][ddims[3]];

        for (int i = 0; i < ddims[0]; i++) {
            Bitmap bitmap = bitmaps[i];
            datasets[i] = MyUtil.normalizeImage(bitmap);
        }
        return datasets;
    }
}

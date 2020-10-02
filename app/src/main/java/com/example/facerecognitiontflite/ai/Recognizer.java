package com.example.facerecognitiontflite.ai;

import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import org.jetbrains.annotations.NotNull;

import java.io.FileDescriptor;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Recognizer {
    public static class Recognition {
        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        Recognition(final Float confidence, final RectF location) {
            this.confidence = confidence;
            this.location = location;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        @SuppressLint("DefaultLocale")
        @NotNull
        @Override
        public String toString() {
            String resultString = "";
//            if (id != null) {
//                resultString += "[" + id + "] ";
//            }
//
//            if (title != null) {
//                resultString += title + " ";
//            }

            if (confidence != null) resultString += String.format("(%.1f%%) ", confidence * 100.0f);

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    private static Recognizer recognizer;

    private BlazeFace blazeFace;
    private MobileFaceNet faceNet;

    private Recognizer() {}

    static Recognizer getInstance (AssetManager assetManager) throws Exception {
        if (recognizer != null) return recognizer;

        recognizer = new Recognizer();
        recognizer.blazeFace = BlazeFace.create(assetManager);
        recognizer.faceNet = MobileFaceNet.create(assetManager);

        return recognizer;
    }

    List<Recognition> recognizeImage(Bitmap bitmap, Matrix matrix) {
        synchronized (this) {
            List<RectF> faces = blazeFace.detect(bitmap);
            final List<Recognition> mappedRecognitions = new LinkedList<>();

            for (RectF rectF : faces) {
                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);

                matrix.mapRect(rectF);

                Recognition result =
                        new Recognition(0f, rectF);
                mappedRecognitions.add(result);
            }
            return mappedRecognitions;
        }
    }

    void updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            ArrayList<float[]> list = new ArrayList<>();

            for (Uri uri : uris) {
                Bitmap bitmap = getBitmapFromUri(contentResolver, uri);
                List<RectF> faces = blazeFace.detect(bitmap);

                Rect rect = new Rect();
                if (!faces.isEmpty()) {
                    faces.get(0).round(rect);
                }

                float[] emb_array = new float[MobileFaceNet.EMBEDDING_SIZE];
                faceNet.getEmbeddings(bitmap, rect).get(emb_array);
                list.add(emb_array);
            }
        }
    }

    private Bitmap getBitmapFromUri(ContentResolver contentResolver, Uri uri) throws Exception {
        ParcelFileDescriptor parcelFileDescriptor =
                contentResolver.openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();

        return bitmap;
    }

    void enableStatLogging(final boolean debug){
    }

    void close() {
        blazeFace.close();
        faceNet.close();
    }
}

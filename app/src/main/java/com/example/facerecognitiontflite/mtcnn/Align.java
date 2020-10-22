package com.example.facerecognitiontflite.mtcnn;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Point;

/**
 * Face Alignment using face landamarks
 */
public class Align {

    /**
     * @param bitmap : input bitmap image
     * @param landmarks : face landmarks
     * @return aligned face
     */
    public static Bitmap face_align(Bitmap bitmap, Point[] landmarks) {
        float diffEyeX = landmarks[1].x - landmarks[0].x;
        float diffEyeY = landmarks[1].y - landmarks[0].y;

        float fAngle;
        if (Math.abs(diffEyeY) < 1e-7) {
            fAngle = 0.f;
        } else {
            fAngle = (float) (Math.atan(diffEyeY / diffEyeX) * 180.0f / Math.PI);
        }
        Matrix matrix = new Matrix();
        matrix.setRotate(-fAngle);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}

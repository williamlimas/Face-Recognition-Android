package com.example.facerecognitiontflite;

import com.example.facerecognitiontflite.mobilefacenet.MobileFaceNet;

public class Person {
    private float[][] embeddings;
    private int idx_write;
    private int idx_read;
    private static final int MAX_IDX = 10;

    public float[] getEmbedding(int idx) { return embeddings[idx]; }

    public int getIdx_read() {
        return idx_read;
    }

    public Person(){
        this.embeddings = new float[MAX_IDX][MobileFaceNet.EMBEDDING_SIZE];
        this.idx_write = 0;
        this.idx_read = 0;
    }

    protected void addEmbedding(float[] embedding){
        this.embeddings[this.idx_write] = embedding;
        this.idx_write = this.idx_write+1;
        if (this.idx_write > MAX_IDX-1) this.idx_write = 0;
        this.idx_read = Math.min(this.idx_read+1, MAX_IDX-1);
    }
}

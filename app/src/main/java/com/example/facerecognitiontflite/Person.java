package com.example.facerecognitiontflite;

import java.util.ArrayList;
import java.util.List;

public class Person {
    private final List<float[]> embeddings;

    public float[] getEmbedding(int idx) { return embeddings.get(idx); }

    public int getEmbeddingSize(){ return this.embeddings.size(); }

    public Person(){
        this.embeddings = new ArrayList<>();
    }

    protected void addEmbedding(float[] embedding){
        this.embeddings.add(embedding);
    }
}

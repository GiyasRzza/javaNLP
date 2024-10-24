package org.example;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class OpenNLPExample {

    public static void main(String[] args) {
        try {
            InputStream tokenModelIn = Files.newInputStream(Paths.get("en-token.bin"));
            TokenizerModel tokenModel = new TokenizerModel(tokenModelIn);
            Tokenizer tokenizer = new TokenizerME(tokenModel);

            InputStream posModelIn = Files.newInputStream(Paths.get("en-pos-maxent.bin"));
            POSModel posModel = new POSModel(posModelIn);
            POSTaggerME posTagger = new POSTaggerME(posModel);

            String sentence = "Apache OpenNLP is a machine learning based toolkit.";

            String[] tokens = tokenizer.tokenize(sentence);

            String[] posTags = posTagger.tag(tokens);

            for (int i = 0; i < tokens.length; i++) {
                System.out.println(tokens[i] + " : " + posTags[i]);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
package org.example;

import com.johnsnowlabs.nlp.*;
import com.johnsnowlabs.nlp.annotators.ner.NerConverter;
public class SparkNLPExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Spark NLP NER Example")
                .master("local[*]")
                .config("spark.driver.memory", "2g")
                .getOrCreate();

        Dataset<Row> data = spark.createDataFrame(new JavaRDD<>(Arrays.asList(
                new GenericRowWithSchema(new Object[]{"Albert Einstein lived in Bern."}),
                new GenericRowWithSchema(new Object[]{"Microsoft is a software company."})
        ), new StructType(new StructField[]{
                new StructField("text", DataTypes.StringType, false)
        })));

        DocumentAssembler docAssembler = new DocumentAssembler()
                .setInputCol("text")
                .setOutputCol("document");

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("document")
                .setOutputCol("token");

        NerDLModel nerModel = NerDLModel.pretrained()
                .setInputCols("document", "token")
                .setOutputCol("ner")
                .setCaseSensitive(true);

        NerConverter converter = new NerConverter()
                .setInputCols("document", "token", "ner")
                .setOutputCol("ner_chunk");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{docAssembler, tokenizer, nerModel, converter});


        Dataset<Row> result = pipeline.fit(data).transform(data);

        result.show(false);

        spark.stop();
    }
}
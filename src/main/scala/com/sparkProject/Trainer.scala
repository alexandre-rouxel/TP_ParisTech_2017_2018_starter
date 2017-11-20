package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ ParamGridBuilder, TrainValidationSplit}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    // only for self made function
    // import spark.implicits._

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   // a) Charger un csv dans dataframe
   val df : DataFrame = spark
     .read
     .parquet("/Users/alexandre/MSBGD/spark/tp/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")

    /** TF-IDF **/
    // stage 1
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //StopWordsRemover
    // stage 2
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // stage 3
    val countvectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vectorized")
      .setMinDF(2)

    // stage 4

    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")


    /** VECTOR ASSEMBLER **/

    // stage 5

    // stream indexer
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    // stage 6
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    /** MODEL **/
    //stage 7

    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign",  "hours_prepa" , "goal" , "country_indexed" , "currency_indexed"))
      .setOutputCol("features")


    //stage 8

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0).setFitIntercept(true) .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array( 0.7, 0.3) )
      .setTol( 1.0e-6)
      .setMaxIter( 300)

    /** PIPELINE **/

    val pipeline = new Pipeline()
      .setStages(Array ( tokenizer , remover ,  countvectorizer , idf , indexer , indexer2 , vecAssembler ,lr))

    /** TRAINING AND GRID-SEARCH **/

    /** build a training set  **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)


    /** define parameters set **/

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Predef.intArrayOps(Array(-8, -2, 2)).map(math.exp(_)))
      .addGrid(countvectorizer.minDF,Array(55.0,95.0,20))
      .build()

    //"evaluatorF1" is not supported by Binary Classifier : BinaryClassificationEvaluator

    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluatorF1)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /** run the cross validator on the training set **/
    val cvModel = cv.fit(training)

    /** run the cross validator on test and training set **/
    val trainPredictions = cvModel.transform(training)
    val testPredictions = cvModel.transform(test)

    /** evaluate the linear classifier on training and test sets**/
    val f1Test = evaluatorF1.evaluate(testPredictions)
    val f1Train = evaluatorF1.evaluate(trainPredictions)


    val df_WithPredictions : DataFrame = testPredictions

    df_WithPredictions.groupBy( "final_status" , "predictions" ).count.show()

    println("F1 measurement on training set  ")
    println(f1Train)
    println("F1 measurement on test set  ")
    println(f1Test)

    /** save the trained model **/
    cvModel.write.overwrite().save("./fittedBinaryClassifier")




  }
}

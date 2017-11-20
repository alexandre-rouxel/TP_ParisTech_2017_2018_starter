package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._


    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/

    // a) Charger un csv dans dataframe
    val df: DataFrame = spark
      .read
      .option("header", true)  // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .csv("/Users/alexandre/MSBGD/spark/tp/TP_ParisTech_2017_2018_starter/data/train.csv")

    // b) nombre de lignes et colonnes
    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    // c) Observer le dataframe: First 20 rows, and all columns :
    df.show()

    // d) Le schema donne le nom et type (string, integer,...) de chaque colonne
    df.printSchema()

    // e) Assigner le bon type aux colonnes
    val dfCasted = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    /** 2 - CLEANING **/

    dfCasted.groupBy("final_status").count.orderBy($"count".desc).show

    // a) Description statistique
    dfCasted.select("goal", "backers_count", "final_status").describe().show

    // b) Observer les autres colonnes, comprendre les données, et proposer des cleanings
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)




    // c) Quelques cleanings possibles

    // enlever la colonne "disable_communication".
    // cette colonne est très largement majoritairement à "false", il y a 311 "true" (négligable) le reste est non-identifié donc:
    val df2: DataFrame = dfCasted.drop("disable_communication")


    // f) LES FUITES DU FUTUR:
    // dans les datasets construits a posteriori des évènements, il arrive que des données ne pouvant être connues qu'après
    // la résolution de chaque évènement soient insérées dans le dataset. On a des fuites depuis le futur !
    //
    // Par exemple, on a ici le nombre de "backers" dans la colonne "backers_count". Il s'agit du nombre de personnes FINAL
    // ayant investi dans chaque projet, or ce nombre n'est connu qu'après la fin de la campagne.
    //
    // Il faut savoir repérer et traiter ces données pour plusieurs raisons:
    //    - En pratique quand on voudra appliquer notre modèle, les données du futur ne sont pas présentes (puisqu'elles ne sont pas encore connues).
    //      On ne peut donc pas les utiliser comme input pour un modèle.
    //    - Pendant l'entraînement (si on ne les a pas enlevées) elles facilitent le travail du modèle puisque qu'elles contiennent
    //      des informations directement liées à ce qu'on veut prédir. Par exemple, si backers_count = 0 on est sûr que la
    //      campagne a raté.

    // d)
    // Pour enlever les données du futur on retir les colonnes "backers_count" et "state_changed_at".
    val dfNoFutur: DataFrame = df2
      .drop("backers_count", "state_changed_at")

    // e)
    // On pourrait penser que "currency" et "country" sont redondantes, auquel cas on pourrait enlever une des colonne.
    // Mais en y regardant de plus près:
    //   - dans la zone euro: même monnaie pour différents pays => garder les deux colonnes.
    //   - il semble y avoir des inversions entre ces deux colonnes et du nettoyage à faire en utilisant les deux colonnes.
    //     En particulier on peut remarquer que quand country=false le country à l'air d'être dans currency:

    df.filter($"country".isNull).groupBy("currency").count.orderBy($"count".desc).show(50)

    def udf_country = udf{(country: String, currency: String) =>
      if (country == null) // && currency != "false")
        currency
      else
        country //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    def udf_currency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }


    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))
      .drop("country", "currency")

    dfCountry.groupBy("country2", "currency2").count.orderBy($"count".desc).show(50)


    // Pour aider notre algorithme, on souhaite qu'un même mot écrit en minuscules ou majuscules ne soit pas deux
    // "entités" différentes. On met tout en minuscules
    val dfLower: DataFrame = dfCountry
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    dfLower.show(50)


    // Remplacer les strings "false" dans currency et country
    // En observant les colonnes
    dfLower.groupBy("country2").count.orderBy($"count".desc).show(100)
    dfLower.groupBy("currency2").count.orderBy($"count".desc).show(100)


    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/

    // a) b) c) features à partir des timestamp
    val dfDurations: DataFrame = dfLower
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2")) // datediff requires a dateType
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3)) // here timestamps are in seconds, there are 3600 seconds in one hour
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at", "deadline", "launched_at")


    // d)

    val dfText= dfDurations
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))


    /** VALEUR NULLES **/
    dfDurations.show(2)
    dfText.show(2)

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))

    // vérifier l'équilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show

    // filtrer les classes qui nous intéressent
    // Final status contient d'autres états que Failed ou Succeed. On ne sait pas ce que sont ces états,
    // on peut les enlever ou les considérer comme Failed également. Seul "null" est ambigue et on les enlève.
    val dfFiltered = dfReady.filter($"final_status".isin(0, 1))

    dfFiltered.show(50)
    println(dfFiltered.count)


    /** WRITING DATAFRAME **/


    // dfFiltered.write.mode(SaveMode.Overwrite).parquet("/Users/maxime/TP_parisTech_2017_2018/data/prepared_trainingset")
    dfFiltered.write.mode(SaveMode.Overwrite).parquet("/Users/alexandre/MSBGD/spark/tp/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")



  }

}

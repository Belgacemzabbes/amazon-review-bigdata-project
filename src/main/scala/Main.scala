package org.belgacem.com

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}


object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")
    // Initialiser SparkSession
    val spark = SparkSession.builder
      .appName("Large Scale Data Exploration")
      .master("local[*]")
      .getOrCreate()

    // Définir le schéma
    val schema = new StructType()
      .add("class", IntegerType, nullable = false)
      .add("title", StringType, nullable = true)
      .add("review", StringType, nullable = true)

    // Charger l'ensemble de données
    val data = loadDataset(spark, "data/train.csv", schema)

    // Exploration des données
    exploreData(spark, data)

    // Analyse avancée
    advancedAnalysis(data)

    // Stop SparkSession
    spark.stop()
  }

  def loadDataset(spark: SparkSession, filePath: String, schema: StructType): DataFrame = {
    spark.read.option("header", "true")
      .option("inferSchema", "true")
      .schema(schema)
      .csv(filePath)
  }

  def exploreData(spark: SparkSession, df: DataFrame): Unit = {
    df.printSchema()
    df.show(5)
    df.describe().show()

    // Compter les lignes
    println(s"Number of rows: ${df.count()}")

    // Requêtes Spark SQL pour des explorations rapides
    df.createOrReplaceTempView("data")
    spark.sql("SELECT COUNT(*) FROM data WHERE class = 1").show()

    // Rechercher des corrélations uniquement pour les colonnes numériques
    val numericColumns = df.schema.fields.filter(_.dataType == IntegerType).map(_.name)
    val correlations = numericColumns.map { col =>
      (col, df.stat.corr("class", col))
    }

  }

  def advancedAnalysis(df: DataFrame): Unit = {
    // Prétraitement des données
    val cleanedData = df.na.fill(0)

    // Convertir les colonnes en vecteurs
    val vectorRDD = cleanedData.rdd.map(row => {
      val features = row.toSeq.toArray.dropRight(1).map(_.toString.toDouble)
      Vectors.dense(features)
    })

    // Calculer les statistiques sommaires
    val summary = Statistics.colStats(vectorRDD)
    println(s"Mean: ${summary.mean}")
    println(s"Variance: ${summary.variance}")
    println(s"NumNonzeros: ${summary.numNonzeros}")

    // Calculer les corrélations
    val correlMatrix = Statistics.corr(vectorRDD)
    println(s"Correlation matrix:\n $correlMatrix")
  }
}
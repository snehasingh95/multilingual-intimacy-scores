import org.apache.spark.sql.SparkSession

object Merge80MData {
  def main(args: Array[String]) {
	val spark:SparkSession = SparkSession.builder()
		  .master("local[1]")
		  .appName("trainAnalysis")
		  .getOrCreate() 
	val movie=spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("movie_questions.csv").drop("_c0").withColumnRenamed("question", "text").withColumnRenamed("intimacy", "label").withColumn("language", lit("English"))
	movie.show(120)
	movie.coalesce(1).write.csv("movies_new.csv")
	System.out.println(spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("movie_questions.csv").count())
	System.out.println(movie.count())
	val book=spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("book_questions.csv").drop("_c0").\
		  withColumnRenamed("question", "text").withColumnRenamed("intimacy", "label").withColumn("language", lit("English"))
	book.show(1)
	book.coalesce(1).write.csv("books_new.csv")
	print(spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("book_questions.csv").count())
	print(book.count())
	val book=spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("book_questions.csv").drop("_c0").withColumnRenamed("question", "text").withColumnRenamed("intimacy", "label").withColumn("language", lit("English"))
	book.show(1)
	book.coalesce(1).write.csv("books_new.csv")
	print(spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("book_questions.csv").count())
	print(book.count())
	val reddit_post=spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("reddit_post_questions.csv").drop("_c0").withColumnRenamed("question", "text").withColumnRenamed("intimacy", "label").withColumn("language", lit("English"))
	reddit_post.show(1)
	reddit_post.coalesce(1).write.csv("reddit_post_new.csv")
	print(spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("reddit_post_questions.csv").count())
	print(reddit_post.count())
	val reddit_comment=spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("reddit_comment_questions.csv").drop("_c0").withColumnRenamed("question", "text").withColumnRenamed("intimacy", "label").withColumn("language", lit("English"))
	reddit_comment.show(1)
	reddit_comment.coalesce(1).write.csv("reddit_comment_new.csv")
	print(spark.read.option("quote", "\"").option("escape", "\"").option("header","true").csv("reddit_comment_questions.csv").count())
	print(reddit_comment.count())
	all = movie.union(book).union(reddit_post).union(reddit_comment)
	val all = movie.union(book).union(reddit_post).union(reddit_comment)
	all.count()
	all.coalesce(1).write.csv("all_new.csv")
	
	spark.stop()
  }
}
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class CoOccurrenceMatrixMapper extends Mapper<Object, Text, Text, IntWritable> {
	@Override
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		//value = userid \t movie1: rating, movie2: rating...
		//key = movie1: movie2 value = 1
		//calculate the movies which are watched by the same user
		//occurrence list: <movieA, movieB>
		String[] user_movieRating = value.toString().trim().split("\t");
		String[] movie_ratings = user_movieRating[1].split(",");

		for (String movieA_rating : movie_ratings) {
			String movieA = movieA_rating.split(":")[0];
			for (String movieB_rating : movie_ratings) {
				String movieB = movieB_rating.split(":")[0];
				context.write(new Text(movieA + ":" + movieB), new IntWritable(1));
			}
		}
	}
}
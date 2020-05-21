import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class UserRatingMatrixMapper extends Mapper<Object, Text, IntWritable, Text> {
	@Override
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		//input user,movie,rating
		//divide data by user
		String[] user_movieRating = value.toString().trim().split(",");
		if (user_movieRating.length != 3){
			return;
		}
		int userID = Integer.parseInt(user_movieRating[0]);
		context.write(new IntWritable(userID), new Text(user_movieRating[1] + ":" + user_movieRating[2]));
	}
}
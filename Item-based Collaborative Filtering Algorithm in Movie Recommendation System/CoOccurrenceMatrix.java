import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class CoOccurrenceMatrix {
	public static void main(String[] args) throws Exception{
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf, "co-occurrence matrix");
		job.setMapperClass(CoOccurrenceMatrixMapper.class);
		job.setReducerClass(CoOccurrenceMatrixReducer.class);
		
		job.setJarByClass(CoOccurrenceMatrix.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		TextInputFormat.setInputPaths(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));
		
		job.waitForCompletion(true);
		
	}
}

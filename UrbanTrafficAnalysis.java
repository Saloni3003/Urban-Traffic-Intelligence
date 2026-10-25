import java.io.IOException;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Locale;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class UrbanTrafficAnalysis {
    private static final DateTimeFormatter[] TIME_FORMATS = {
        DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss", Locale.ROOT),
        DateTimeFormatter.ofPattern("M/d/uuuu H:mm", Locale.ROOT),
        DateTimeFormatter.ofPattern("M/d/uuuu H:mm:ss", Locale.ROOT),
        DateTimeFormatter.ISO_LOCAL_DATE_TIME
    };

    // CSV parser that respects commas inside quoted fields and escaped double quotes.
    static String[] parseCSV(String line) {
        java.util.ArrayList<String> fields = new java.util.ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    field.append('"');
                    i++;
                } else {
                    quoted = !quoted;
                }
            } else if (c == ',' && !quoted) {
                fields.add(field.toString());
                field.setLength(0);
            } else {
                field.append(c);
            }
        }
        fields.add(field.toString());
        return fields.toArray(new String[0]);
    }

    static String category(String value) {
        String v = value == null ? "" : value.trim();
        return v.isEmpty() ? "UNKNOWN" : v.toUpperCase(Locale.ROOT);
    }

    static Integer parseHour(String value) {
        String v = value == null ? "" : value.trim();
        if (v.isEmpty()) return null;
        try { return OffsetDateTime.parse(v).getHour(); }
        catch (DateTimeParseException ignored) { }
        try { return LocalDateTime.parse(v, DateTimeFormatter.ISO_LOCAL_DATE_TIME).getHour(); }
        catch (DateTimeParseException ignored) { }
        for (DateTimeFormatter formatter : TIME_FORMATS) {
            try { return LocalDateTime.parse(v, formatter).getHour(); }
            catch (DateTimeParseException ignored) { }
        }
        return null;
    }

    public static class TrafficMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private static final LongWritable ONE = new LongWritable(1);
        private final Text outputKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            if (key.get() == 0 && line.startsWith("ID,")) return;
            if (line.trim().isEmpty()) return;

            String[] f = parseCSV(line);
            // This dataset has 46 columns, indexed 0 through 45.
            if (f.length < 46) {
                context.getCounter("TrafficAnalysis", "MALFORMED_ROWS").increment(1);
                return;
            }

            // Correct zero-based column indexes from the dataset header:
            // Severity=2, Start_Time=3, State=14, Weather_Condition=28, Sunrise_Sunset=42.
            emit(context, "STATE", category(f[14]));
            emit(context, "SEVERITY", category(f[2]));
            emit(context, "WEATHER", category(f[28]));

            String sun = f[42] == null ? "" : f[42].trim().toUpperCase(Locale.ROOT);
            if (!sun.isEmpty()) {
                String period = sun.contains("NIGHT") ? "NIGHT" :
                    sun.contains("DAY") ? "DAY" : "UNKNOWN";
                emit(context, "DAY_NIGHT", period);
            }

            Integer hour = parseHour(f[3]);
            if (hour != null && hour >= 0 && hour <= 23) {
                emit(context, "HOUR", String.format(Locale.ROOT, "%02d", hour));
            } else {
                context.getCounter("TrafficAnalysis", "UNPARSEABLE_START_TIME").increment(1);
            }
            context.getCounter("TrafficAnalysis", "VALID_ROWS").increment(1);
        }

        private void emit(Context context, String group, String value)
                throws IOException, InterruptedException {
            outputKey.set(group + "|" + value);
            context.write(outputKey, ONE);
        }
    }

    public static class TrafficReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private final LongWritable result = new LongWritable();

        @Override
        protected void reduce(Text key, Iterable<LongWritable> values, Context context)
                throws IOException, InterruptedException {
            long sum = 0;
            for (LongWritable value : values) sum += value.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: UrbanTrafficAnalysis <input-path> <output-path>");
            System.exit(2);
        }
        Job job = Job.getInstance(new Configuration(),
                "Urban Traffic Intelligence - MapReduce Aggregation");
        job.setJarByClass(UrbanTrafficAnalysis.class);
        job.setMapperClass(TrafficMapper.class);
        job.setCombinerClass(TrafficReducer.class);
        job.setReducerClass(TrafficReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

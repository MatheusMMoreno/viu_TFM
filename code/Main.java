import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

public class Main {
    public static void main(String[] args) {
        // Call your Python script for face recognition
        try {
            // Adjust the file path to point to the correct directory
            String pythonScriptPath = "C:\\Users\\mathe\\StudioProjects\\tfm_viu\\src\\assets\\face_recognition_script.py";
            Process process = Runtime.getRuntime().exec("python " + pythonScriptPath);

            // Monitor CPU and memory usage of the Java application
            Runtime runtime = Runtime.getRuntime();
            long startTime = System.currentTimeMillis();
            ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
            long startCpuTime = threadMXBean.getCurrentThreadCpuTime();
            long startMemoryUsage = runtime.totalMemory() - runtime.freeMemory();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                // Print the output from the Python script
                System.out.println(line);
            }
            // Wait for the process to finish
            process.waitFor();

            // Print any errors that occurred during execution
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                System.err.println(errorLine);
            }

            // Measure resource usage
            long elapsedTime = System.currentTimeMillis() - startTime;
            long endCpuTime = threadMXBean.getCurrentThreadCpuTime();
            long endMemoryUsage = runtime.totalMemory() - runtime.freeMemory();
            long cpuTimeUsed = endCpuTime - startCpuTime;
            long memoryUsed = endMemoryUsage - startMemoryUsage;

            System.out.println("Elapsed Time: " + elapsedTime + " ms");
            System.out.println("CPU Time Used: " + cpuTimeUsed + " ns");
            System.out.println("Memory Used: " + memoryUsed + " bytes");
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}

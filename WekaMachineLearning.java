//Copyright Daniel G. Perkins, 2017

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
 
public class WekaMachineLearning {
	//this method reads the data file using a buffered reader
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	// this method evaluates the training and testing data in order to determine accuracy
	// The evaluation class has the evaluateModel() method that allows it to summarize the algorithm's learning process
	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
	
	//this method calculates the accuracy of the testing portion of the algorithm based on the actual data and the predicted values of the algorithm
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	//This splits the data into testing and training data
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Random rand = new Random(1);
        data.randomize(rand);

		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		int fold;
		//default filename is iris.arff
		String filename = "";
		Scanner sc = new Scanner(System.in);
		
		//In order to create a more efficient code, I did not have the user determine the algorithm, but instead I ran all three algorithms together
		//This allows the user to compare the algorithms easier
		
		//gets the file the user wants to read from 
		System.out.println("Which file would you like to read from:"
				+ "\n1. Iris.arff"
				+ "\n2. Diabetes.arff"
				+ "\n3. Soybean.arff");
		while (true) {
			int file = sc.nextInt();
			if (file == 1) {
				filename = "iris.arff";
				break;
			}
			if (file == 2) {

				filename = "diabetes.arff";
				break;
			}
			if (file == 3) {
				filename = "soybean.arff";
				break;
			}
			if (file > 3 || file < 1) {
				System.out.println("ERROR: Please enter a valid number");
			}
		}
			
		
		//gets number of desired k folds from user input
		System.out.println("How many folds?");
		while (true) {
			fold = sc.nextInt();
			if (fold < 2) {
				System.out.println("ERROR: Please enter an integer greater than or equal to 2");
			}
			if (fold >= 2) {
				int trainingPercent = (fold - 1) * 100 / fold;
				int testingPercent = 100 - trainingPercent;
				System.out.println("A k Value of " + fold + " will result in " + trainingPercent +  "% training and " + testingPercent + "% testing.");
				break;
			}
		}
		System.out.println("=================================");
		
		//reads file
		BufferedReader datafile = readDataFile(filename);
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
        data.stratify(fold);

        
		// Do cross validation with k fold based on user input
		Instances[][] split = crossValidationSplit(data, fold);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		// Use a set of classifiers
		Classifier[] models = { 
				new J48(), 
				new PART(), 
				new JRip(),
		};
 
		// Run for each model
		for (int j = 0; j < models.length; j++) {
			// Collect every group of predictions for current model in a FastVector
			//http://weka.sourceforge.net/doc.dev/weka/core/FastVector.html
			FastVector predictions = new FastVector();
 
			// For each training-testing split pair, train and test the classifier
			for (int i = 0; i < trainingSplits.length; i++) {
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
 
				predictions.appendElements(validation.predictions());
 
				
			}
 
			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);
			
			// Print current classifier's name and accuracy
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");
		}
 
	}
}
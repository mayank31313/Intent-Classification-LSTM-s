package ai.mayank;

import org.bytedeco.javacpp.Loader;
import org.datavec.api.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;

//Created By Mayank
public class Main{

    public static void main(String[] args) throws Exception {
        int iterations = 1000;
        //Word2Vec vectorModel = WordVectorSerializer.readWord2VecModel("M:\\DeepLearning\\dl4j-examples-master\\dl4j-examples\\src\\main\\resources\\NewsData\\NewsWordVector.txt");


        Word2Vec vectorModel = WordVectorSerializer.readWord2VecModel("M:\\Artificial Intelligence\\Datasets\\pruned.word2vec.txt");
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        vectorModel.setTokenizerFactory(tokenizerFactory);
        //RecordIterator recordsIterator = new RecordIterator(vectorModel,50,256);
        CSVRecordIterator recordsIterator = new CSVRecordIterator(vectorModel,50,256,"X:\\languageanalysis1.csv");
        int inputNeurons = recordsIterator.inputColumns();
        int outputNeurons = recordsIterator.getLabels().size();
        int nEpochs = 1;
        MultiLayerNetwork net;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(iterations)
            .updater(Updater.RMSPROP)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(0.01)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(inputNeurons).nOut(256)
                .activation(Activation.SOFTSIGN).build())
            //.layer(1,new GravesLSTM.Builder().nIn(200).nOut(300).activation("softsign").build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(outputNeurons).build())
            .pretrain(false).backprop(true).build();

        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener());
        Evaluation eval = new Evaluation();
        System.out.println("Starting training");
        for(int i=0;i<nEpochs;i++) {
            net.fit(recordsIterator);
            recordsIterator.reset();
            while(recordsIterator.hasNext()) {
                DataSet set = recordsIterator.next(1);
                INDArray outputLabels = net.output(set.getFeatureMatrix());
                eval.eval(set.getLabels().getRow(0),outputLabels.getRow(0));
            }
        }
        System.out.println(eval.stats());

        File model = new File("X:\\languageAnalysis.model");
        if(model.exists()) model.delete();
        ModelSerializer.writeModel(net, model,true);
        //net = ModelSerializer.restoreMultiLayerNetwork("X:\\languageAnalysis.model");
        Scanner s = new Scanner(System.in);
        String read="";
        System.out.println("Enter Query: ");
        while(!(read = s.nextLine()).equals("")) {
            INDArray output = net.output(recordsIterator.loadfeaturesFromString(read)).getRow(0);
            List<Pair<Number,String>> pair = new ArrayList<>();
            for (int i = 0; i < output.rows(); i++) {
                System.out.println(output.getRow(i) + "    ****    " + recordsIterator.getLabels().get(i));
                pair.add(Pair.makePair(output.getRow(i).meanNumber(),recordsIterator.getLabels().get(i)));
            }
            Collections.sort(pair, new Comparator<Pair<Number, String>>() {
                @Override
                public int compare(Pair<Number, String> o1, Pair<Number, String> o2) {
                    if(o1.getFirst().doubleValue()<o2.getFirst().doubleValue()) return 1;
                    else return -1;
                }
            });
            System.out.println(pair.get(0).getSecond() + pair.get(0).getFirst());
        }
    }
}

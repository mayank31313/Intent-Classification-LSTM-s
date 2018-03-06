package ai.mayank;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Created by mayan on 3/5/2018.
 */
public class CSVRecordIterator implements DataSetIterator{
    private Word2Vec wordVector;
    private TokenizerFactory tokenizerFactory;
    private int inputNeurons;
    private int outputNeurons;
    private List<String> labels;
    private int cursor = 0;
    private int batchSize;
    private int truncateLength = 0;
    final List<Pair<String, List<String>>> pair = new ArrayList<>();

    public CSVRecordIterator(Word2Vec word2Vec, int batchSize, int truncateLength,String csvPath) throws IOException,InterruptedException{
        RecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(new File(csvPath)));
        TreeSet<String> intents = new TreeSet<>();
        this.wordVector = word2Vec;
        this.tokenizerFactory = wordVector.getTokenizerFactory();
        while(reader.hasNext()){
            List<Writable> split = (List<Writable>) reader.next();
            String query = split.get(0).toString();
            String label = split.get(1).toString();
            System.out.println(split);

            if (label.equals("None")) continue;

            List<String> tokens = this.tokenizerFactory.create(query).getTokens();
            List<String> filteredTokens = new ArrayList<>();
            for (String token : tokens) {
                if (wordVector.hasWord(token)) filteredTokens.add(token);
            }
            if (filteredTokens.size() > 1) {
                Pair<String, List<String>> p = Pair.newPair(label, filteredTokens);
                pair.add(p);
                intents.add(label);
            }
        }

        reader.close();
        labels = new ArrayList<>(intents);
        Collections.shuffle(pair);
        outputNeurons = getLabels().size();
        inputNeurons = wordVector.getWordVector(wordVector.getLookupTable().getVocabCache().elementAtIndex(0).getWord()).length;
        this.batchSize = batchSize;
        this.truncateLength = truncateLength;
    }

    @Override
    public DataSet next(int num) {
        List<Pair<String, List<String>>> queries = new ArrayList<>(num);

        for (int i = 0; i < num && cursor < numExamples(); i++) {
            Pair<String, List<String>> container = pair.get(cursor);
            queries.add(container);

            cursor++;
        }
        int maxLength = 0;
        List<List<String>> allTokens = new ArrayList<>();
        for (Pair<String, List<String>> p : queries) {
            List<String> tokens = p.getSecond();
            allTokens.add(tokens);
            maxLength = Math.max(maxLength, tokens.size());
        }
        if (maxLength > truncateLength) maxLength = truncateLength;

        INDArray features = Nd4j.create(queries.size(), inputNeurons, maxLength);
        INDArray labels = Nd4j.create(queries.size(), getLabels().size(), maxLength);

        INDArray featuresMask = Nd4j.zeros(queries.size(), maxLength);     // Create Matrix of queries.size() * maxLength
        INDArray labelsMask = Nd4j.zeros(queries.size(), maxLength);


        for (int i = 0; i < allTokens.size(); i++) {
            List<String> tokens = allTokens.get(i);
            int seqLength = Math.min(tokens.size(), maxLength);
            for (int j = 0; j < tokens.size(); j++) {
                final INDArray vectors = wordVector.getWordVectorMatrix(tokens.get(j));
                features.put(new INDArrayIndex[]{
                    point(i), all(), point(j)
                }, vectors);
                featuresMask.putScalar(new int[]{i, j}, 1);
            }

            int idx = getLabels().indexOf(queries.get(i).getFirst());
            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
        }


        DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
        return ds;
    }

    @Override
    public int totalExamples() {
        return pair.size();
    }

    @Override
    public int inputColumns() {
        return inputNeurons;
    }

    @Override
    public int totalOutcomes() {
        return outputNeurons;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    public INDArray loadfeaturesFromString(String text) {
        List<String> tokens = tokenizerFactory.create(text).getTokens();
        List<String> filteredTokens = new ArrayList<>();
        for (String token : tokens) {
            if (wordVector.hasWord(token))
                filteredTokens.add(token);
        }
        INDArray features = Nd4j.create(1, inputNeurons, filteredTokens.size());
        for (int i = 0; i < filteredTokens.size(); i++) {
            INDArray vector = wordVector.getWordVectorMatrix(filteredTokens.get(i));
            features.put(new INDArrayIndex[]{
                point(0), all(), point(i)
            }, vector);
        }
        return features;
    }
}

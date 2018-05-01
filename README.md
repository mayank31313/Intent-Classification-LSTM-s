# Intent-Classification-with-Deep-Learning

Hello there,
    Deep Learning for Java <a href="https://deeplearning4j.org/">link</a><br>
    Word2Vectors <a href="https://drive.google.com/open?id=1216vROaeWkTvtkeAdJi-21uW93kTrHAP">Click Here...</a>
    
    Word2Vec vectorModel = WordVectorSerializer.readWord2VecModel("M:\\Artificial Intelligence\\Datasets\\word2vec.txt"); // Word2Vec Path
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        vectorModel.setTokenizerFactory(tokenizerFactory);
        //RecordIterator recordsIterator = new RecordIterator(vectorModel,50,256);
        CSVRecordIterator recordsIterator = new CSVRecordIterator(vectorModel,50,256,"X:\\languageanalysis1.csv");//CSV Path

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

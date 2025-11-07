import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class GRUModel {
  constructor(inputShape, outputSize) {
    this.model = null;
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.history = null;
  }

  buildModel() {
    this.model = tf.sequential();
    
    // First GRU layer
    this.model.add(tf.layers.gru({
      units: 64,
      returnSequences: true,
      inputShape: this.inputShape
    }));
    
    // Second GRU layer
    this.model.add(tf.layers.gru({
      units: 32,
      returnSequences: false
    }));
    
    // Dropout for regularization
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // Output layer - 10 stocks × 3 days = 30 binary outputs
    this.model.add(tf.layers.dense({
      units: this.outputSize,
      activation: 'sigmoid'
    }));

    // Compile model
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    return this.model;
  }

  async train(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32) {
    this.history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: [X_test, y_test],
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
        }
      }
    });
    
    return this.history;
  }

  async predict(X) {
    return this.model.predict(X);
  }

  evaluate(X_test, y_test) {
    return this.model.evaluate(X_test, y_test);
  }

  computePerStockAccuracy(predictions, y_true, symbols, predictionDays) {
    const predArray = predictions.arraySync();
    const trueArray = y_true.arraySync();
    const numStocks = symbols.length;
    
    const accuracies = {};
    
    symbols.forEach((symbol, stockIdx) => {
      let correct = 0;
      let total = 0;
      
      for (let i = 0; i < predArray.length; i++) {
        for (let day = 0; day < predictionDays; day++) {
          const predIdx = stockIdx * predictionDays + day;
          const predicted = predArray[i][predIdx] > 0.5 ? 1 : 0;
          const actual = trueArray[i][predIdx];
          
          if (predicted === actual) {
            correct++;
          }
          total++;
        }
      }
      
      accuracies[symbol] = total > 0 ? correct / total : 0;
    });
    
    return accuracies;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
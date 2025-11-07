const tf = globalThis.tf;

if (!tf || typeof tf.sequential !== 'function') {
  throw new Error('TensorFlow.js failed to initialise. Ensure tf.min.js loads before gru.js.');
}

export class GRUModel {
  constructor({ inputShape, outputSize }) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.model = null;
    this.history = null;
  }

  build({ dropoutRate = 0.2, gruUnits = [96, 64] } = {}) {
    if (!Array.isArray(gruUnits) || gruUnits.length === 0) {
      throw new Error('gruUnits must be a non-empty array');
    }

    const model = tf.sequential();

    gruUnits.forEach((units, idx) => {
      model.add(tf.layers.gru({
        units,
        returnSequences: idx !== gruUnits.length - 1,
        dropout: 0.1,
        recurrentDropout: 0.1,
        inputShape: idx === 0 ? this.inputShape : undefined,
      }));
    });

    if (dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: dropoutRate }));
    }

    model.add(tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });

    this.model = model;
    return this.model;
  }

  async train({
    X_train,
    y_train,
    X_val,
    y_val,
    epochs = 40,
    batchSize = 32,
    onEpochEnd,
  }) {
    if (!this.model) {
      throw new Error('Call build() before training the model.');
    }

    this.history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: false,
      validationData: X_val && y_val ? [X_val, y_val] : undefined,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (typeof onEpochEnd === 'function') {
            onEpochEnd(epoch, logs);
          }
          await tf.nextFrame();
        },
      },
    });

    return this.history;
  }

  predict(X) {
    if (!this.model) {
      throw new Error('Model is not built.');
    }
    return this.model.predict(X);
  }

  evaluate(X, y) {
    if (!this.model) {
      throw new Error('Model is not built.');
    }
    return this.model.evaluate(X, y);
  }

  analysePredictions(predictionsArray, groundTruthArray, symbols, predictionDays) {
    const perStockTimeline = {};
    const perStockTotals = {};
    const perStockConfusion = {};
    let globalCorrect = 0;
    let globalTotal = 0;

    symbols.forEach((symbol) => {
      perStockTimeline[symbol] = [];
      perStockTotals[symbol] = { correct: 0, total: 0 };
      perStockConfusion[symbol] = { tp: 0, fp: 0, fn: 0, tn: 0 };
    });

    predictionsArray.forEach((row, rowIndex) => {
      symbols.forEach((symbol, stockIdx) => {
        const offset = stockIdx * predictionDays;
        let correctCount = 0;
        const perDayFlags = [];

        for (let day = 0; day < predictionDays; day += 1) {
          const pred = row[offset + day] >= 0.5 ? 1 : 0;
          const actual = groundTruthArray[rowIndex][offset + day];

          if (pred === 1 && actual === 1) {
            perStockConfusion[symbol].tp += 1;
          } else if (pred === 1 && actual === 0) {
            perStockConfusion[symbol].fp += 1;
          } else if (pred === 0 && actual === 1) {
            perStockConfusion[symbol].fn += 1;
          } else {
            perStockConfusion[symbol].tn += 1;
          }

          if (pred === actual) {
            correctCount += 1;
            globalCorrect += 1;
          }
          globalTotal += 1;
          perDayFlags.push({ predicted: pred, actual, correct: pred === actual });
        }

        perStockTimeline[symbol].push({
          correctCount,
          total: predictionDays,
          perDayFlags,
        });

        perStockTotals[symbol].correct += correctCount;
        perStockTotals[symbol].total += predictionDays;
      });
    });

    const perStockAccuracy = {};
    symbols.forEach((symbol) => {
      const { correct, total } = perStockTotals[symbol];
      perStockAccuracy[symbol] = total === 0 ? 0 : correct / total;
    });

    const overallAccuracy = globalTotal === 0 ? 0 : globalCorrect / globalTotal;

    return {
      perStockAccuracy,
      perStockTimeline,
      perStockConfusion,
      overallAccuracy,
    };
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

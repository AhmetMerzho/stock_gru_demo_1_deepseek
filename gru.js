const tf = globalThis.tf;

if (!tf || typeof tf.sequential !== 'function') {
  throw new Error('TensorFlow.js failed to initialise. Ensure tf.min.js loads before gru.js.');
}

function getCurrentLearningRate(optimizer) {
  if (!optimizer) {
    return null;
  }

  if (typeof optimizer.getLearningRate === 'function') {
    const lr = optimizer.getLearningRate();
    if (typeof lr === 'number') {
      return lr;
    }
    if (lr && typeof lr.dataSync === 'function') {
      const values = lr.dataSync();
      return values.length > 0 ? values[0] : null;
    }
  }

  if (typeof optimizer.learningRate === 'number') {
    return optimizer.learningRate;
  }

  return null;
}

function setLearningRate(optimizer, value) {
  if (!optimizer || !Number.isFinite(value)) {
    return;
  }

  if (typeof optimizer.setLearningRate === 'function') {
    optimizer.setLearningRate(value);
  } else {
    optimizer.learningRate = value;
  }
}

const HAS_LAYER_NORM = typeof tf.layers.layerNormalization === 'function';

function createReduceLROnPlateauCallback(optimizer, {
  monitor = 'val_loss',
  factor = 0.5,
  patience = 3,
  minDelta = 1e-4,
  minLR = 1e-5,
} = {}) {
  let best = Number.POSITIVE_INFINITY;
  let wait = 0;

  return new tf.CustomCallback({
    onEpochEnd: async (epoch, logs) => {
      const current = logs?.[monitor];
      if (current == null || !Number.isFinite(current)) {
        await tf.nextFrame();
        return;
      }

      if (current < best - minDelta) {
        best = current;
        wait = 0;
      } else {
        wait += 1;
        if (wait >= patience) {
          wait = 0;
          const lr = getCurrentLearningRate(optimizer);
          if (lr !== null) {
            const next = Math.max(lr * factor, minLR);
            if (next < lr - 1e-12) {
              setLearningRate(optimizer, next);
              console.log(`ReduceLROnPlateau: reducing learning rate to ${next.toExponential(2)}`);
            }
          }
        }
      }

      await tf.nextFrame();
    },
  });
}

export class GRUModel {
  constructor({ inputShape, outputSize }) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.model = null;
    this.history = null;
    this.optimizer = null;
    this.initialLearningRate = 0.001;
  }

  build({
    dropoutRate = 0.25,
    gruUnits = [128, 96, 64],
    denseUnits = [128, 64],
    bidirectional = true,
    learningRate = 0.001,
    recurrentDropout = 0.1,
    convFilters = [96],
    convKernelSize = 3,
    convActivation = 'relu',
    convDropout = 0.1,
  } = {}) {
    if (!Array.isArray(gruUnits) || gruUnits.length === 0) {
      throw new Error('gruUnits must be a non-empty array');
    }

    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    if (this.optimizer && typeof this.optimizer.dispose === 'function') {
      this.optimizer.dispose();
    }
    this.optimizer = null;

    const model = tf.sequential();

    const hasConvStack = Array.isArray(convFilters) && convFilters.length > 0;
    if (hasConvStack) {
      convFilters.forEach((filters, idx) => {
        if (!Number.isFinite(filters) || filters <= 0) {
          throw new Error('convFilters must contain positive numbers');
        }

        const kernel = Array.isArray(convKernelSize)
          ? convKernelSize[idx % convKernelSize.length]
          : convKernelSize;

        const convConfig = {
          filters,
          kernelSize: Number.isFinite(kernel) && kernel > 0 ? kernel : 3,
          padding: 'causal',
          activation: convActivation,
          kernelInitializer: 'heNormal',
        };

        if (idx === 0) {
          convConfig.inputShape = this.inputShape;
        }

        model.add(tf.layers.conv1d(convConfig));

        if (HAS_LAYER_NORM) {
          model.add(tf.layers.layerNormalization());
        }

        if (convDropout > 0) {
          model.add(tf.layers.dropout({ rate: Math.min(0.5, convDropout) }));
        }
      });
    }

    gruUnits.forEach((units, idx) => {
      const gruConfig = {
        units,
        returnSequences: idx !== gruUnits.length - 1,
        dropout: Math.min(0.4, dropoutRate),
        recurrentDropout: Math.min(0.4, recurrentDropout),
      };

      if (!bidirectional && idx === 0 && !hasConvStack) {
        gruConfig.inputShape = this.inputShape;
      }

      const gruLayer = tf.layers.gru(gruConfig);

      if (bidirectional) {
        const wrapperConfig = {
          layer: gruLayer,
          mergeMode: 'concat',
        };

        if (idx === 0 && !hasConvStack) {
          wrapperConfig.inputShape = this.inputShape;
        }

        model.add(tf.layers.bidirectional(wrapperConfig));
      } else {
        model.add(gruLayer);
      }

      if (HAS_LAYER_NORM) {
        model.add(tf.layers.layerNormalization());
      }
    });

    if (dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: Math.min(0.5, dropoutRate) }));
    }

    if (Array.isArray(denseUnits) && denseUnits.length > 0) {
      denseUnits.forEach((units, index) => {
        if (!Number.isFinite(units) || units <= 0) {
          throw new Error('denseUnits must contain positive numbers');
        }

        model.add(
          tf.layers.dense({
            units,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            kernelRegularizer: tf.regularizers.l2({ l2: 1e-5 }),
          }),
        );

        if (dropoutRate > 0 && index !== denseUnits.length - 1) {
          model.add(tf.layers.dropout({ rate: Math.min(0.5, dropoutRate * 1.1) }));
        }
      });
    }

    model.add(tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }));

    this.optimizer = tf.train.adam(learningRate);
    this.initialLearningRate = learningRate;

    model.compile({
      optimizer: this.optimizer,
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
    epochs = 60,
    batchSize = 32,
    onEpochEnd,
    earlyStoppingPatience,
    reduceLrPatience,
    reduceLrFactor = 0.5,
    minLearningRate = 1e-5,
  } = {}) {
    if (!this.model) {
      throw new Error('Call build() before training the model.');
    }

    const hasValidation = Boolean(X_val && y_val);
    const callbacks = [];

    callbacks.push(
      new tf.CustomCallback({
        onBatchEnd: async () => {
          await tf.nextFrame();
        },
        onEpochEnd: async (epoch, logs) => {
          if (typeof onEpochEnd === 'function') {
            onEpochEnd(epoch, logs);
          }
          await tf.nextFrame();
        },
      }),
    );

    if (hasValidation) {
      const patience = earlyStoppingPatience ?? Math.max(8, Math.round(epochs * 0.15));
      callbacks.push(
        tf.callbacks.earlyStopping({
          monitor: 'val_loss',
          patience,
          minDelta: 1e-4,
        }),
      );

      const reducePatience = reduceLrPatience ?? Math.max(4, Math.round(patience / 2));
      const factor = Math.min(0.9, Math.max(0.1, reduceLrFactor ?? 0.5));

      callbacks.push(
        createReduceLROnPlateauCallback(this.optimizer, {
          monitor: 'val_loss',
          factor,
          patience: reducePatience,
          minDelta: 1e-4,
          minLR: minLearningRate ?? 1e-5,
        }),
      );
    }

    this.history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: false,
      validationData: hasValidation ? [X_val, y_val] : undefined,
      callbacks,
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

  getLearningRate() {
    return getCurrentLearningRate(this.optimizer);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    if (this.optimizer && typeof this.optimizer.dispose === 'function') {
      this.optimizer.dispose();
    }
    this.optimizer = null;
    this.history = null;
  }
}

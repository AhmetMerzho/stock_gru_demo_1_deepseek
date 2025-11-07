const tf = globalThis.tf;

if (!tf || typeof tf.tensor !== 'function') {
  throw new Error('TensorFlow.js failed to initialise. Ensure tf.min.js loads before data-loader.js.');
}

const REQUIRED_COLUMNS = ['Date', 'Symbol', 'Open', 'Close'];

const FEATURE_KEYS = ['Open', 'Close', 'Return', 'Momentum3', 'Volatility5'];

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

function normaliseDateString(dateString) {
  const value = new Date(dateString);
  if (Number.isNaN(value.getTime())) {
    throw new Error(`Invalid date encountered: ${dateString}`);
  }
  return value.toISOString().slice(0, 10);
}

function fillMissingValues(array) {
  let lastValid = null;
  for (let i = 0; i < array.length; i += 1) {
    const value = array[i];
    if (Number.isFinite(value)) {
      lastValid = value;
    } else if (lastValid !== null) {
      array[i] = lastValid;
    }
  }

  lastValid = null;
  for (let i = array.length - 1; i >= 0; i -= 1) {
    const value = array[i];
    if (Number.isFinite(value)) {
      lastValid = value;
    } else if (lastValid !== null) {
      array[i] = lastValid;
    }
  }

  return array;
}

function minMaxScale(series) {
  const filtered = series.filter((v) => Number.isFinite(v));
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return series.map(() => 0);
  }
  return series.map((value) => ((value - min) / (max - min)));
}

function standardScale(series) {
  const filtered = series.filter((v) => Number.isFinite(v));
  if (filtered.length === 0) {
    return series.map(() => 0);
  }

  const mean = filtered.reduce((acc, value) => acc + value, 0) / filtered.length;
  const variance = filtered.reduce((acc, value) => acc + (value - mean) ** 2, 0) / filtered.length;
  const std = Math.sqrt(variance);

  if (!Number.isFinite(std) || std === 0) {
    return series.map(() => 0);
  }

  return series.map((value) => (Number.isFinite(value) ? (value - mean) / std : 0));
}

export class DataLoader {
  constructor({ sequenceLength = 12, predictionHorizon = 3, trainSplit = 0.8 } = {}) {
    this.sequenceLength = sequenceLength;
    this.predictionHorizon = predictionHorizon;
    this.trainSplit = trainSplit;

    this.rows = [];
    this.symbols = [];
    this.dates = [];
    this.featureCube = {};
    this.normalisedCube = {};
    this.featuresPerSymbol = FEATURE_KEYS.slice();
  }

  reset() {
    this.rows = [];
    this.symbols = [];
    this.dates = [];
    this.featureCube = {};
    this.normalisedCube = {};
    this.featuresPerSymbol = FEATURE_KEYS.slice();
  }

  async loadFile(file) {
    if (!(file instanceof File)) {
      throw new Error('Expected a File object for CSV upload.');
    }

    const text = await file.text();
    this.parseCSV(text);
  }

  async loadFromUrl(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Unable to fetch sample CSV (${response.status} ${response.statusText}).`);
    }

    const text = await response.text();
    this.parseCSV(text);
  }

  parseCSV(csvText) {
    const lines = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter((line) => line.trim().length > 0);
    if (lines.length <= 1) {
      throw new Error('CSV file is empty.');
    }

    const headers = parseCSVLine(lines[0]);
    const missing = REQUIRED_COLUMNS.filter((col) => !headers.includes(col));
    if (missing.length > 0) {
      throw new Error(`CSV is missing required columns: ${missing.join(', ')}`);
    }

    const columnIndex = headers.reduce((acc, header, index) => ({ ...acc, [header]: index }), {});

    this.rows = [];
    for (let i = 1; i < lines.length; i += 1) {
      const values = parseCSVLine(lines[i]);
      if (values.length !== headers.length) {
        continue;
      }

      const date = normaliseDateString(values[columnIndex.Date]);
      const symbol = values[columnIndex.Symbol];
      const open = Number.parseFloat(values[columnIndex.Open]);
      const close = Number.parseFloat(values[columnIndex.Close]);

      if (!symbol) {
        continue;
      }

      this.rows.push({ date, symbol, open, close });
    }

    if (this.rows.length === 0) {
      throw new Error('No valid data rows were parsed from the CSV.');
    }

    this.rows.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.symbol.localeCompare(b.symbol)));
  }

  buildFeatureCube() {
    const dateSet = new Set();
    const symbolSet = new Set();

    this.rows.forEach(({ date, symbol }) => {
      dateSet.add(date);
      symbolSet.add(symbol);
    });

    this.dates = Array.from(dateSet).sort((a, b) => new Date(a) - new Date(b));
    this.symbols = Array.from(symbolSet).sort();

    this.featureCube = {};
    this.symbols.forEach((symbol) => {
      this.featureCube[symbol] = {};
      FEATURE_KEYS.forEach((key) => {
        this.featureCube[symbol][key] = new Array(this.dates.length).fill(Number.NaN);
      });
    });

    const dateIndex = this.dates.reduce((acc, date, idx) => ({ ...acc, [date]: idx }), {});

    this.rows.forEach(({ date, symbol, open, close }) => {
      const idx = dateIndex[date];
      const target = this.featureCube[symbol];
      if (!target) {
        return;
      }
      target.Open[idx] = Number.isFinite(open) ? open : Number.NaN;
      target.Close[idx] = Number.isFinite(close) ? close : Number.NaN;
    });

    this.symbols.forEach((symbol) => {
      ['Open', 'Close'].forEach((feature) => {
        fillMissingValues(this.featureCube[symbol][feature]);
      });

      const closes = this.featureCube[symbol].Close;
      const returns = closes.map((value, index) => {
        if (index === 0) {
          return 0;
        }
        const previous = closes[index - 1];
        if (!Number.isFinite(value) || !Number.isFinite(previous) || previous === 0) {
          return 0;
        }
        return (value - previous) / previous;
      });

      const momentum3 = closes.map((value, index) => {
        if (index < 3) {
          return 0;
        }
        const baseline = closes[index - 3];
        if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline === 0) {
          return 0;
        }
        return (value - baseline) / baseline;
      });

      const volatility5 = closes.map((_, index) => {
        if (index < 4) {
          return 0;
        }
        const window = closes.slice(index - 4, index + 1);
        if (window.some((v) => !Number.isFinite(v))) {
          return 0;
        }
        const mean = window.reduce((acc, v) => acc + v, 0) / window.length;
        if (!Number.isFinite(mean) || mean === 0) {
          return 0;
        }
        const variance = window.reduce((acc, v) => acc + (v - mean) ** 2, 0) / window.length;
        const std = Math.sqrt(variance);
        if (!Number.isFinite(std)) {
          return 0;
        }
        return std / mean;
      });

      this.featureCube[symbol].Return = returns;
      this.featureCube[symbol].Momentum3 = momentum3;
      this.featureCube[symbol].Volatility5 = volatility5;
    });

    this.normalisedCube = {};
    this.symbols.forEach((symbol) => {
      this.normalisedCube[symbol] = {
        Open: minMaxScale(this.featureCube[symbol].Open.slice()),
        Close: minMaxScale(this.featureCube[symbol].Close.slice()),
        Return: standardScale(this.featureCube[symbol].Return.slice()),
        Momentum3: standardScale(this.featureCube[symbol].Momentum3.slice()),
        Volatility5: standardScale(this.featureCube[symbol].Volatility5.slice()),
      };
    });
  }

  createWindowedDataset() {
    const sequences = [];
    const targets = [];
    const sampleDates = [];

    const { sequenceLength, predictionHorizon, featuresPerSymbol } = this;
    const numSymbols = this.symbols.length;

    for (let anchor = sequenceLength - 1; anchor < this.dates.length - predictionHorizon; anchor += 1) {
      const sequence = [];
      let valid = true;

      for (let offset = sequenceLength - 1; offset >= 0; offset -= 1) {
        const index = anchor - offset;
        const timestep = [];

        for (let s = 0; s < numSymbols; s += 1) {
          const symbol = this.symbols[s];

          for (let f = 0; f < featuresPerSymbol.length; f += 1) {
            const featureKey = featuresPerSymbol[f];
            const value = this.normalisedCube[symbol][featureKey][index];

            if (!Number.isFinite(value)) {
              valid = false;
              break;
            }

            timestep.push(value);
          }

          if (!valid) {
            break;
          }
        }

        if (!valid) {
          break;
        }

        sequence.push(timestep);
      }

      if (!valid) {
        continue;
      }

      const targetVector = [];
      const baselineIndex = anchor;

      for (let s = 0; s < numSymbols; s += 1) {
        const symbol = this.symbols[s];
        const baselineClose = this.featureCube[symbol].Close[baselineIndex];
        if (!Number.isFinite(baselineClose)) {
          valid = false;
          break;
        }

        for (let horizon = 1; horizon <= predictionHorizon; horizon += 1) {
          const futureClose = this.featureCube[symbol].Close[baselineIndex + horizon];
          if (!Number.isFinite(futureClose)) {
            valid = false;
            break;
          }
          targetVector.push(futureClose > baselineClose ? 1 : 0);
        }

        if (!valid) {
          break;
        }
      }

      if (!valid) {
        continue;
      }

      sequences.push(sequence);
      targets.push(targetVector);
      sampleDates.push(this.dates[anchor]);
    }

    if (sequences.length === 0) {
      throw new Error('Unable to construct any sequences. Please provide a longer time series.');
    }

    const featuresPerStep = numSymbols * featuresPerSymbol.length;
    const X = tf.tensor(sequences, [sequences.length, sequenceLength, featuresPerStep], 'float32');
    const y = tf.tensor(targets, [targets.length, numSymbols * this.predictionHorizon], 'float32');

    if (sequences.length < 2) {
      X.dispose();
      y.dispose();
      throw new Error('Not enough samples to create a train/test split.');
    }

    const rawSplit = Math.floor(sequences.length * this.trainSplit);
    const splitIndex = Math.min(sequences.length - 1, Math.max(1, rawSplit));
    const X_train = X.slice([0, 0, 0], [splitIndex, sequenceLength, featuresPerStep]);
    const X_test = X.slice([splitIndex, 0, 0], [sequences.length - splitIndex, sequenceLength, featuresPerStep]);
    const y_train = y.slice([0, 0], [splitIndex, numSymbols * this.predictionHorizon]);
    const y_test = y.slice([splitIndex, 0], [targets.length - splitIndex, numSymbols * this.predictionHorizon]);

    X.dispose();
    y.dispose();

    return {
      X_train,
      X_test,
      y_train,
      y_test,
      sampleDates,
      splitIndex,
    };
  }

  async prepareDataset() {
    this.buildFeatureCube();
    const dataset = this.createWindowedDataset();

    return {
      ...dataset,
      symbols: this.symbols.slice(),
      sequenceLength: this.sequenceLength,
      predictionDays: this.predictionHorizon,
      featuresPerSymbol: this.featuresPerSymbol.slice(),
      totalDates: this.dates.length,
      totalRows: this.rows.length,
    };
  }
}

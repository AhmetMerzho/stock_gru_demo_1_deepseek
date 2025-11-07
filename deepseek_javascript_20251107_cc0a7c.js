import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class DataLoader {
  constructor() {
    this.rawData = [];
    this.processedData = [];
    this.symbols = new Set();
    this.dateIndex = new Map();
    this.featureData = {};
    this.normalizedData = {};
    this.X = null;
    this.y = null;
    this.trainTestSplit = 0.8;
  }

  async loadCSV(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csv = e.target.result;
        this.parseCSV(csv);
        resolve();
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    
    this.rawData = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const row = {};
      headers.forEach((header, index) => {
        row[header.trim()] = values[index]?.trim();
      });
      this.rawData.push(row);
    }
  }

  preprocessData() {
    // Group by symbol and create date index
    this.symbols = new Set(this.rawData.map(row => row.Symbol));
    const allDates = [...new Set(this.rawData.map(row => row.Date))].sort();
    
    // Create date index map
    this.dateIndex = new Map(allDates.map((date, idx) => [date, idx]));
    
    // Initialize feature data structure
    this.featureData = {};
    this.symbols.forEach(symbol => {
      this.featureData[symbol] = {
        Open: new Array(allDates.length).fill(null),
        Close: new Array(allDates.length).fill(null)
      };
    });

    // Fill feature data
    this.rawData.forEach(row => {
      const dateIdx = this.dateIndex.get(row.Date);
      const symbol = row.Symbol;
      if (this.featureData[symbol]) {
        this.featureData[symbol].Open[dateIdx] = parseFloat(row.Open);
        this.featureData[symbol].Close[dateIdx] = parseFloat(row.Close);
      }
    });

    // Forward fill missing values
    this.symbols.forEach(symbol => {
      ['Open', 'Close'].forEach(feature => {
        let lastValue = null;
        for (let i = 0; i < allDates.length; i++) {
          if (this.featureData[symbol][feature][i] !== null) {
            lastValue = this.featureData[symbol][feature][i];
          } else if (lastValue !== null) {
            this.featureData[symbol][feature][i] = lastValue;
          }
        }
      });
    });

    // Normalize data per symbol
    this.normalizedData = {};
    this.symbols.forEach(symbol => {
      this.normalizedData[symbol] = {
        Open: this.minMaxNormalize(this.featureData[symbol].Open),
        Close: this.minMaxNormalize(this.featureData[symbol].Close)
      };
    });

    return this.createSequences();
  }

  minMaxNormalize(data) {
    const filtered = data.filter(val => val !== null);
    const min = Math.min(...filtered);
    const max = Math.max(...filtered);
    return data.map(val => (val - min) / (max - min));
  }

  createSequences() {
    const sequenceLength = 12;
    const predictionDays = 3;
    const numStocks = this.symbols.size;
    const symbolsArray = Array.from(this.symbols);
    
    const sequences = [];
    const targets = [];

    // Create mapping from date index to actual date for debugging
    const dateArray = Array.from(this.dateIndex.keys());

    for (let i = sequenceLength; i < dateArray.length - predictionDays; i++) {
      const sequence = [];
      
      // Create input sequence (12 days of Open, Close for all stocks)
      for (let j = i - sequenceLength; j < i; j++) {
        const timeStep = [];
        symbolsArray.forEach(symbol => {
          timeStep.push(
            this.normalizedData[symbol].Open[j],
            this.normalizedData[symbol].Close[j]
          );
        });
        sequence.push(timeStep);
      }

      // Create target (binary classification for next 3 days for each stock)
      const target = [];
      symbolsArray.forEach(symbol => {
        const currentClose = this.featureData[symbol].Close[i];
        for (let k = 1; k <= predictionDays; k++) {
          const futureClose = this.featureData[symbol].Close[i + k];
          if (futureClose !== null && currentClose !== null) {
            target.push(futureClose > currentClose ? 1 : 0);
          } else {
            target.push(0); // Default to 0 if missing data
          }
        }
      });

      sequences.push(sequence);
      targets.push(target);
    }

    // Convert to tensors
    this.X = tf.tensor3d(sequences);
    this.y = tf.tensor2d(targets);

    // Split into train/test
    const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
    
    const X_train = this.X.slice([0, 0, 0], [splitIndex, -1, -1]);
    const X_test = this.X.slice([splitIndex, 0, 0], [-1, -1, -1]);
    const y_train = this.y.slice([0, 0], [splitIndex, -1]);
    const y_test = this.y.slice([splitIndex, 0], [-1, -1]);

    return {
      X_train,
      X_test,
      y_train,
      y_test,
      symbols: symbolsArray,
      dateArray,
      sequenceLength,
      predictionDays
    };
  }

  dispose() {
    if (this.X) this.X.dispose();
    if (this.y) this.y.dispose();
  }
}
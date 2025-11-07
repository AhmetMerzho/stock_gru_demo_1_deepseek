import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model = null;
    this.trainingData = null;
    this.isTraining = false;
    this.charts = {};
    
    this.initializeUI();
  }

  initializeUI() {
    // File input handler
    document.getElementById('csvFile').addEventListener('change', (e) => {
      this.handleFileUpload(e.target.files[0]);
    });

    // Train button handler
    document.getElementById('trainBtn').addEventListener('click', () => {
      this.trainModel();
    });

    // Initialize charts container
    this.initializeCharts();
  }

  async handleFileUpload(file) {
    if (!file) return;
    
    try {
      document.getElementById('status').textContent = 'Loading CSV file...';
      await this.dataLoader.loadCSV(file);
      
      document.getElementById('status').textContent = 'Preprocessing data...';
      this.trainingData = this.dataLoader.preprocessData();
      
      document.getElementById('status').textContent = 'Data loaded successfully!';
      document.getElementById('trainBtn').disabled = false;
      
      console.log('Data shapes:', {
        X_train: this.trainingData.X_train.shape,
        X_test: this.trainingData.X_test.shape,
        y_train: this.trainingData.y_train.shape,
        y_test: this.trainingData.y_test.shape
      });
      
    } catch (error) {
      document.getElementById('status').textContent = `Error: ${error.message}`;
      console.error('File loading error:', error);
    }
  }

  async trainModel() {
    if (!this.trainingData || this.isTraining) return;
    
    this.isTraining = true;
    document.getElementById('trainBtn').disabled = true;
    
    try {
      // Build model
      const inputShape = [this.trainingData.sequenceLength, this.trainingData.symbols.length * 2];
      const outputSize = this.trainingData.symbols.length * this.trainingData.predictionDays;
      
      this.model = new GRUModel(inputShape, outputSize);
      this.model.buildModel();
      
      document.getElementById('status').textContent = 'Training model...';
      
      // Train model
      await this.model.train(
        this.trainingData.X_train,
        this.trainingData.y_train,
        this.trainingData.X_test,
        this.trainingData.y_test,
        50, // epochs
        32   // batchSize
      );
      
      // Evaluate model
      document.getElementById('status').textContent = 'Evaluating model...';
      const predictions = await this.model.predict(this.trainingData.X_test);
      const accuracies = this.model.computePerStockAccuracy(
        predictions,
        this.trainingData.y_test,
        this.trainingData.symbols,
        this.trainingData.predictionDays
      );
      
      // Display results
      this.displayResults(accuracies, predictions);
      
      document.getElementById('status').textContent = 'Training completed!';
      
      // Clean up
      predictions.dispose();
      
    } catch (error) {
      document.getElementById('status').textContent = `Training error: ${error.message}`;
      console.error('Training error:', error);
    } finally {
      this.isTraining = false;
      document.getElementById('trainBtn').disabled = false;
    }
  }

  displayResults(accuracies, predictions) {
    // Sort stocks by accuracy
    const sortedStocks = Object.entries(accuracies)
      .sort(([, a], [, b]) => b - a);
    
    // Update accuracy bar chart
    this.updateAccuracyChart(sortedStocks);
    
    // Update prediction timeline
    this.updatePredictionTimeline(predictions, sortedStocks);
    
    // Display accuracy table
    this.updateAccuracyTable(sortedStocks);
  }

  updateAccuracyChart(sortedStocks) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    // Destroy previous chart if exists
    if (this.charts.accuracy) {
      this.charts.accuracy.destroy();
    }
    
    this.charts.accuracy = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: sortedStocks.map(([symbol]) => symbol),
        datasets: [{
          label: 'Prediction Accuracy',
          data: sortedStocks.map(([, accuracy]) => accuracy * 100),
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }]
      },
      options: {
        indexAxis: 'y',
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'Accuracy (%)'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Stock Prediction Accuracy Ranking'
          }
        }
      }
    });
  }

  updatePredictionTimeline(predictions, sortedStocks) {
    // This is a simplified timeline visualization
    // In a real implementation, you'd want to show correct/incorrect predictions over time
    const ctx = document.getElementById('timelineChart').getContext('2d');
    
    // Destroy previous chart if exists
    if (this.charts.timeline) {
      this.charts.timeline.destroy();
    }
    
    // For demo purposes, show average prediction confidence over time
    const predData = predictions.arraySync();
    const avgConfidence = predData.map(pred => 
      pred.reduce((sum, val) => sum + val, 0) / pred.length
    );
    
    this.charts.timeline = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({length: avgConfidence.length}, (_, i) => `Day ${i + 1}`),
        datasets: [{
          label: 'Average Prediction Confidence',
          data: avgConfidence,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          tension: 0.1,
          fill: true
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            title: {
              display: true,
              text: 'Confidence'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Prediction Confidence Timeline'
          }
        }
      }
    });
  }

  updateAccuracyTable(sortedStocks) {
    const table = document.getElementById('accuracyTable');
    table.innerHTML = '';
    
    // Create header
    const header = table.createTHead();
    const headerRow = header.insertRow();
    headerRow.insertCell().textContent = 'Stock';
    headerRow.insertCell().textContent = 'Accuracy';
    headerRow.insertCell().textContent = 'Rank';
    
    // Create body
    const tbody = table.createTBody();
    sortedStocks.forEach(([symbol, accuracy], index) => {
      const row = tbody.insertRow();
      row.insertCell().textContent = symbol;
      row.insertCell().textContent = `${(accuracy * 100).toFixed(2)}%`;
      row.insertCell().textContent = index + 1;
    });
  }

  initializeCharts() {
    // Initialize empty charts
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    
    this.charts.accuracy = new Chart(accuracyCtx, {
      type: 'bar',
      data: { labels: [], datasets: [] },
      options: { indexAxis: 'y' }
    });
    
    this.charts.timeline = new Chart(timelineCtx, {
      type: 'line',
      data: { labels: [], datasets: [] }
    });
  }

  dispose() {
    if (this.dataLoader) {
      this.dataLoader.dispose();
    }
    if (this.model) {
      this.model.dispose();
    }
    Object.values(this.charts).forEach(chart => chart.destroy());
  }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.stockApp = new StockPredictionApp();
});
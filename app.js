import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.trainingData = null;
    this.model = null;
    this.accuracyChart = null;

    this.dom = {
      fileInput: document.getElementById('csvFile'),
      loadSampleBtn: document.getElementById('loadSampleBtn'),
      clearBtn: document.getElementById('clearBtn'),
      trainBtn: document.getElementById('trainBtn'),
      statusPanel: document.getElementById('statusPanel'),
      statusMessage: document.getElementById('statusMessage'),
      statusDetail: document.getElementById('statusDetail'),
      datasetSummary: document.getElementById('datasetSummary'),
      epochInput: document.getElementById('epochInput'),
      batchInput: document.getElementById('batchInput'),
      trainingStatus: document.getElementById('trainingStatus'),
      trainingMessage: document.getElementById('trainingMessage'),
      trainingProgress: document.getElementById('trainingProgress'),
      accuracyTableBody: document.getElementById('accuracyTableBody'),
      timelineContainer: document.getElementById('timelineContainer'),
      confusionContainer: document.getElementById('confusionContainer'),
    };

    this.sampleDatasetUrl = './data/sp500_top10_xcorr_recent3y.csv';

    this.attachEventListeners();
    this.initCharts();
    this.setTrainButtonEnabled(false);
  }

  attachEventListeners() {
    this.dom.fileInput.addEventListener('change', async (event) => {
      const [file] = event.target.files;
      if (file) {
        await this.loadDatasetFromFile(file);
      }
    });

    this.dom.loadSampleBtn.addEventListener('click', async () => {
      await this.loadDatasetFromSample();
    });

    this.dom.clearBtn.addEventListener('click', () => {
      this.resetApplicationState();
    });

    this.dom.trainBtn.addEventListener('click', async () => {
      await this.handleTrain();
    });
  }

  initCharts() {
    const context = document.getElementById('accuracyChart').getContext('2d');
    this.accuracyChart = new Chart(context, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Accuracy (%)',
            data: [],
            backgroundColor: 'rgba(59, 130, 246, 0.7)',
            borderColor: 'rgba(29, 78, 216, 0.9)',
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: (value) => `${value}%`,
            },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.formattedValue}% accuracy`,
            },
          },
        },
      },
    });
  }

  setStatus(message, detail = '') {
    this.dom.statusMessage.textContent = message;
    this.dom.statusDetail.textContent = detail;
  }

  setTrainingStatus(visible, message = '', detail = '') {
    this.dom.trainingStatus.hidden = !visible;
    this.dom.trainingMessage.textContent = message;
    this.dom.trainingProgress.textContent = detail;
  }

  async loadDatasetFromFile(file) {
    try {
      this.setStatus('Reading CSV file…');
      await this.dataLoader.loadFile(file);
      await this.prepareDataset();
    } catch (error) {
      console.error(error);
      this.handleError(error);
    }
  }

  async loadDatasetFromSample() {
    try {
      this.setStatus('Fetching bundled dataset…');
      await this.dataLoader.loadFromUrl(this.sampleDatasetUrl);
      await this.prepareDataset(true);
    } catch (error) {
      console.error(error);
      this.handleError(error);
    }
  }

  async prepareDataset(isSample = false) {
    this.setStatus('Preparing tensors…');

    this.disposeTrainingData();
    this.trainingData = await this.dataLoader.prepareDataset();

    const featureCount = this.trainingData.featuresPerSymbol?.length || 0;
    const symbols = this.trainingData.symbols.length;
    this.setStatus(
      'Dataset ready.',
      `${symbols} symbols × ${this.trainingData.sequenceLength}-day windows · ${featureCount} features per symbol.`,
    );
    this.populateDatasetSummary(isSample);
    this.setTrainButtonEnabled(true);
  }

  populateDatasetSummary(isSample) {
    const { datasetSummary } = this.dom;
    datasetSummary.hidden = false;
    datasetSummary.innerHTML = '';

    const featureCount = this.trainingData.featuresPerSymbol?.length || 0;

    const featureNames = (this.trainingData.featuresPerSymbol || []).join(', ') || '—';

    const entries = [
      { title: 'Unique symbols', value: this.trainingData.symbols.length },
      { title: 'Timeline days', value: this.dataLoader.dates.length },
      { title: 'Training samples', value: this.trainingData.splitIndex },
      { title: 'Test samples', value: this.trainingData.sampleDates.length - this.trainingData.splitIndex },
      { title: 'Sequence length', value: `${this.trainingData.sequenceLength} days` },
      { title: 'Prediction horizon', value: `${this.trainingData.predictionDays} days` },
      { title: 'Features per symbol', value: featureCount },
      { title: 'Feature set', value: featureNames },
    ];

    if (isSample) {
      entries.unshift({ title: 'Source', value: 'Bundled sample (S&P 500 subset)' });
    }

    entries.forEach(({ title, value }) => {
      const card = document.createElement('div');
      card.className = 'metric-card';
      card.innerHTML = `<h3>${title}</h3><p>${value}</p>`;
      datasetSummary.appendChild(card);
    });
  }

  async handleTrain() {
    if (!this.trainingData) {
      return;
    }

    const epochs = Number.parseInt(this.dom.epochInput.value, 10) || 60;
    const batchSize = Number.parseInt(this.dom.batchInput.value, 10) || 32;

    try {
      this.setTrainButtonEnabled(false);
      this.setTrainingStatus(true, 'Initialising model…');

      if (this.model) {
        this.model.dispose();
      }

      const featuresPerSymbol = this.trainingData.featuresPerSymbol.length;
      const inputShape = [
        this.trainingData.sequenceLength,
        this.trainingData.symbols.length * featuresPerSymbol,
      ];
      const outputSize = this.trainingData.symbols.length * this.trainingData.predictionDays;

      this.model = new GRUModel({ inputShape, outputSize });
      this.model.build();

      let lastLoss = Number.NaN;
      await this.model.train({
        X_train: this.trainingData.X_train,
        y_train: this.trainingData.y_train,
        X_val: this.trainingData.X_test,
        y_val: this.trainingData.y_test,
        epochs,
        batchSize,
        onEpochEnd: (epoch, logs) => {
          lastLoss = logs.loss;
          const valAcc = logs.val_binaryAccuracy ?? logs.val_acc ?? logs.val_binaryaccuracy;
          const trainAcc = logs.binaryAccuracy ?? logs.acc ?? logs.binaryaccuracy;
          const lr = this.model?.getLearningRate?.();
          const accText = [
            trainAcc !== undefined ? `train acc ${(trainAcc * 100).toFixed(1)}%` : null,
            valAcc !== undefined ? `val acc ${(valAcc * 100).toFixed(1)}%` : null,
          ]
            .filter(Boolean)
            .join(' · ');
          const detailParts = [
            `loss ${logs.loss.toFixed(4)}`,
            accText,
            Number.isFinite(lr) ? `lr ${lr.toExponential(2)}` : null,
          ].filter(Boolean);

          this.setTrainingStatus(true, `Epoch ${epoch + 1}/${epochs}`, detailParts.join(' · '));
        },
      });

      this.setTrainingStatus(true, 'Finalising…', `last loss ${lastLoss.toFixed(4)}`);
      await this.evaluateModel();
    } catch (error) {
      console.error(error);
      this.handleError(error);
    } finally {
      this.setTrainButtonEnabled(true);
      this.setTrainingStatus(false);
    }
  }

  async evaluateModel() {
    if (!this.model || !this.trainingData) {
      return;
    }

    this.setStatus('Evaluating on held-out set…');

    const predictionTensor = this.model.predict(this.trainingData.X_test);
    const [predictionsArray, labelsArray] = await Promise.all([
      predictionTensor.array(),
      this.trainingData.y_test.array(),
    ]);

    predictionTensor.dispose();

    const metrics = this.model.analysePredictions(
      predictionsArray,
      labelsArray,
      this.trainingData.symbols,
      this.trainingData.predictionDays,
    );

    const testSampleDates = this.trainingData.sampleDates.slice(this.trainingData.splitIndex);

    this.renderAccuracyChart(metrics.perStockAccuracy);
    this.renderAccuracyTable(metrics.perStockAccuracy, metrics.perStockConfusion);
    this.renderTimelines(metrics.perStockTimeline, testSampleDates);
    this.renderConfusion(metrics.perStockConfusion);

    this.setStatus('Evaluation complete.', `Overall accuracy ${(metrics.overallAccuracy * 100).toFixed(2)}% on test split.`);
  }

  renderAccuracyChart(perStockAccuracy) {
    const sorted = Object.entries(perStockAccuracy).sort((a, b) => b[1] - a[1]);
    this.accuracyChart.data.labels = sorted.map(([symbol]) => symbol);
    this.accuracyChart.data.datasets[0].data = sorted.map(([, value]) => Number((value * 100).toFixed(2)));
    this.accuracyChart.update();
  }

  renderAccuracyTable(perStockAccuracy, confusion) {
    const sorted = Object.entries(perStockAccuracy)
      .map(([symbol, accuracy]) => ({ symbol, accuracy, confusion: confusion[symbol] }))
      .sort((a, b) => b.accuracy - a.accuracy);

    this.dom.accuracyTableBody.innerHTML = '';

    sorted.forEach(({ symbol, accuracy, confusion: conf }, index) => {
      const row = document.createElement('tr');
      const accuracyPct = (accuracy * 100).toFixed(2);
      const tagClass = accuracy >= 0.66 ? 'success' : accuracy >= 0.45 ? 'warning' : 'danger';

      row.innerHTML = `
        <td>${index + 1}</td>
        <td>${symbol}</td>
        <td><span class="tag ${tagClass}">${accuracyPct}%</span></td>
        <td>${conf.tp} / ${conf.fp} / ${conf.fn} / ${conf.tn}</td>
      `;

      this.dom.accuracyTableBody.appendChild(row);
    });
  }

  renderTimelines(perStockTimeline, sampleDates) {
    const container = this.dom.timelineContainer;
    container.innerHTML = '';

    Object.entries(perStockTimeline).forEach(([symbol, timeline]) => {
      const row = document.createElement('div');
      row.className = 'timeline-row';

      const label = document.createElement('div');
      label.className = 'timeline-label';
      label.textContent = symbol;

      const cells = document.createElement('div');
      cells.className = 'timeline-cells';

      timeline.forEach((item, index) => {
        const cell = document.createElement('div');
        cell.className = 'timeline-cell';

        const { correctCount, total, perDayFlags } = item;
        const isPerfect = correctCount === total;
        const isZero = correctCount === 0;
        cell.classList.add(isPerfect ? 'correct' : isZero ? 'incorrect' : 'partial');

        const date = sampleDates[index] ?? `Sample ${index + 1}`;
        const breakdown = perDayFlags
          .map((flag, idx) => `D+${idx + 1}: ${flag.predicted === 1 ? '↑' : '↓'} (actual ${flag.actual === 1 ? '↑' : '↓'})`)
          .join(' • ');
        cell.dataset.tooltip = `${date}\n${correctCount}/${total} correct\n${breakdown}`;

        cells.appendChild(cell);
      });

      row.appendChild(label);
      row.appendChild(cells);
      container.appendChild(row);
    });
  }

  renderConfusion(perStockConfusion) {
    const container = this.dom.confusionContainer;
    container.innerHTML = '';

    Object.entries(perStockConfusion).forEach(([symbol, conf]) => {
      const card = document.createElement('div');
      card.className = 'confusion-card';
      card.innerHTML = `
        <h4>${symbol}</h4>
        <ul>
          <li>True Positives: ${conf.tp}</li>
          <li>False Positives: ${conf.fp}</li>
          <li>False Negatives: ${conf.fn}</li>
          <li>True Negatives: ${conf.tn}</li>
        </ul>
      `;
      container.appendChild(card);
    });
  }

  handleError(error) {
    this.setStatus('Something went wrong.', error.message || String(error));
    this.setTrainButtonEnabled(false);
  }

  resetApplicationState() {
    this.dataLoader.reset();
    this.disposeTrainingData();
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }

    this.trainingData = null;
    this.setTrainButtonEnabled(false);
    this.dom.statusMessage.textContent = 'Awaiting CSV upload.';
    this.dom.statusDetail.textContent = '';
    this.dom.datasetSummary.hidden = true;
    this.dom.datasetSummary.innerHTML = '';
    this.dom.accuracyTableBody.innerHTML = '';
    this.dom.timelineContainer.innerHTML = '';
    this.dom.confusionContainer.innerHTML = '';

    this.accuracyChart.data.labels = [];
    this.accuracyChart.data.datasets[0].data = [];
    this.accuracyChart.update();

    this.dom.fileInput.value = '';
  }

  disposeTrainingData() {
    if (!this.trainingData) {
      return;
    }

    ['X_train', 'X_test', 'y_train', 'y_test'].forEach((key) => {
      const tensor = this.trainingData[key];
      if (tensor) {
        tensor.dispose();
      }
    });
  }

  setTrainButtonEnabled(enabled) {
    if (!this.dom.trainBtn) {
      return;
    }

    if (enabled) {
      this.dom.trainBtn.disabled = false;
      this.dom.trainBtn.removeAttribute('disabled');
      this.dom.trainBtn.setAttribute('aria-disabled', 'false');
    } else {
      this.dom.trainBtn.disabled = true;
      this.dom.trainBtn.setAttribute('disabled', '');
      this.dom.trainBtn.setAttribute('aria-disabled', 'true');
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.stockApp = new StockPredictionApp();
});

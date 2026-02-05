// charts.js
document.addEventListener('DOMContentLoaded', function() {
    // Only init if canvas exists
    var ctx1 = document.getElementById('performanceChart');
    if (ctx1) {
        new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'Ensemble'],
                datasets: [{
                    label: 'Accuracy',
                    data: [0.85, 0.88, 0.92, 0.91, 0.94], // Placeholder data, strictly for UI demo as requested
                    backgroundColor: [
                        'rgba(231, 76, 60, 0.7)',
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(241, 196, 15, 0.7)'
                    ],
                    borderColor: 'rgba(255,255,255,0.8)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#444' },
                        ticks: { color: '#ddd' }
                    },
                    x: {
                        grid: { color: '#444' },
                        ticks: { color: '#ddd' }
                    }
                },
                plugins: {
                    legend: { labels: { color: 'white' } }
                }
            }
        });
    }
    
    // Comparison Chart Logic potentially dynamically loaded in Reports page
});

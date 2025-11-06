#!/bin/bash

echo "======================================================================="
echo "Starting Merlion Dashboard with Fresh Code (including MOIRAI)"
echo "======================================================================="
echo ""

# Kill any existing dashboard processes
echo "1. Checking for running dashboard processes..."
EXISTING_PID=$(ps aux | grep "merlion.dashboard" | grep -v grep | awk '{print $2}')
if [ ! -z "$EXISTING_PID" ]; then
    echo "   Found existing process (PID: $EXISTING_PID). Stopping it..."
    kill $EXISTING_PID
    sleep 2
    echo "   ✅ Stopped old process"
else
    echo "   ✅ No existing process found"
fi

# Clear Python cache
echo ""
echo "2. Clearing Python cache..."
cd /home/sriteja/Research/Merlion-SF-IIIT
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ Cache cleared"

# Verify MOIRAI is available
echo ""
echo "3. Verifying MOIRAI integration..."
python -c "
from merlion.dashboard.models.forecast import ForecastModel
algorithms = ForecastModel.get_available_algorithms()
if 'Moirai' in algorithms:
    print('   ✅ MOIRAI is available in the algorithms list')
    print(f'   Position: #{algorithms.index(\"Moirai\") + 1} out of {len(algorithms)}')
else:
    print('   ❌ ERROR: MOIRAI not found!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: MOIRAI verification failed!"
    exit 1
fi

# Start dashboard
echo ""
echo "4. Starting dashboard..."
echo "   Dashboard will be available at: http://localhost:8050"
echo ""
echo "======================================================================="
echo "IMPORTANT STEPS TO SEE MOIRAI IN THE DROPDOWN:"
echo "======================================================================="
echo "1. Open http://localhost:8050 in your browser"
echo "2. Navigate to the 'Forecast' page"
echo "3. Upload/select your CSV file"
echo "4. Click on the 'Select Target Variable' dropdown and choose your target"
echo "5. **CLICK ON** the 'Select Forecasting Algorithm' dropdown"
echo "   (Don't just hover - you must click it to populate the options)"
echo "6. MOIRAI should appear as the last option (position #12)"
echo ""
echo "If you still don't see it, press Ctrl+Shift+R in your browser to"
echo "clear the browser cache and reload the page."
echo "======================================================================="
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Start the dashboard
cd /home/sriteja/Research/Merlion-SF-IIIT
python -m merlion.dashboard

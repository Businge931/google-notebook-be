// Netlify Function for FastAPI backend
const { spawn } = require('child_process');
const path = require('path');

exports.handler = async function(event, context) {
  // Set environment variables for production
  process.env.DEBUG = 'false';
  process.env.STORAGE_TYPE = 'local';
  process.env.UPLOAD_DIR = '/tmp/uploads';

  // Path to the Python executable and script
  const pythonPath = process.env.PYTHON_PATH || 'python3';
  const scriptPath = path.resolve(__dirname, '../../main.py');

  try {
    // Run the Python script as a child process
    const pythonProcess = spawn(pythonPath, [scriptPath, JSON.stringify(event)]);

    // Collect data from the Python script
    let result = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    // Wait for the Python script to exit
    const exitCode = await new Promise((resolve, reject) => {
      pythonProcess.on('close', resolve);
      pythonProcess.on('error', reject);
    });

    if (exitCode !== 0) {
      console.error('Python script error:', errorOutput);
      return {
        statusCode: 500,
        body: JSON.stringify({
          status: 'error',
          message: 'Internal server error',
          details: errorOutput
        }),
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': '*',
          'Access-Control-Allow-Headers': '*'
        }
      };
    }

    // Parse the result from the Python script
    let response;
    try {
      response = JSON.parse(result);
    } catch (e) {
      console.error('Failed to parse Python script output:', e);
      return {
        statusCode: 500,
        body: JSON.stringify({
          status: 'error',
          message: 'Failed to parse server response',
          details: e.message
        }),
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': '*',
          'Access-Control-Allow-Headers': '*'
        }
      };
    }

    return {
      statusCode: 200,
      body: JSON.stringify(response),
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': '*',
        'Access-Control-Allow-Headers': '*'
      }
    };
  } catch (error) {
    console.error('Error executing Python script:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({
        status: 'error',
        message: 'Internal server error',
        details: error.message
      }),
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': '*',
        'Access-Control-Allow-Headers': '*'
      }
    };
  }
};

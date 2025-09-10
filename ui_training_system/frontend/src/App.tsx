import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  Alert,
  Snackbar,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Settings,
  Timeline,
  Storage
} from '@mui/icons-material';

import TrainingDashboard from './components/TrainingDashboard';
import TrainingConfig from './components/TrainingConfig';
import TrainingHistory from './components/TrainingHistory';
import ModelManager from './components/ModelManager';
import { TrainingService } from './services/TrainingService';
import { TrainingStatus } from './types/Training';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  const [currentView, setCurrentView] = useState<'dashboard' | 'config' | 'history' | 'models'>('dashboard');
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    experiment_id: null,
    status: 'idle',
    current_iteration: 0,
    total_iterations: 0,
    current_phase: 'idle',
    progress_percentage: 0,
    current_step: 0,
    total_steps: 0,
    loss: null,
    learning_rate: null,
    eta_seconds: null,
    error_message: null,
    model_paths: null
  });
  
  const [configOpen, setConfigOpen] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, message: '', severity: 'info' });

  const trainingService = new TrainingService();

  useEffect(() => {
    // Initialize WebSocket connection
    trainingService.connect(
      (status) => {
        setTrainingStatus(status);
      },
      (message) => {
        console.log('Training log:', message);
      },
      (error) => {
        setNotification({
          open: true,
          message: `Connection error: ${error}`,
          severity: 'error'
        });
      }
    );

    // Load initial status
    loadTrainingStatus();

    return () => {
      trainingService.disconnect();
    };
  }, []);

  const loadTrainingStatus = async () => {
    try {
      const status = await trainingService.getStatus();
      setTrainingStatus(status);
    } catch (error) {
      console.error('Failed to load training status:', error);
    }
  };

  const handleStartTraining = async (config: any) => {
    try {
      const response = await trainingService.startTraining(config);
      if (response.success) {
        setNotification({
          open: true,
          message: 'Training started successfully!',
          severity: 'success'
        });
        setConfigOpen(false);
      }
    } catch (error: any) {
      setNotification({
        open: true,
        message: `Failed to start training: ${error.message}`,
        severity: 'error'
      });
    }
  };

  const handlePauseTraining = async () => {
    try {
      await trainingService.pauseTraining();
      setNotification({
        open: true,
        message: 'Training paused',
        severity: 'info'
      });
    } catch (error: any) {
      setNotification({
        open: true,
        message: `Failed to pause training: ${error.message}`,
        severity: 'error'
      });
    }
  };

  const handleResumeTraining = async () => {
    try {
      await trainingService.resumeTraining();
      setNotification({
        open: true,
        message: 'Training resumed',
        severity: 'info'
      });
    } catch (error: any) {
      setNotification({
        open: true,
        message: `Failed to resume training: ${error.message}`,
        severity: 'error'
      });
    }
  };

  const handleStopTraining = async () => {
    try {
      await trainingService.stopTraining();
      setNotification({
        open: true,
        message: 'Training stop requested',
        severity: 'warning'
      });
    } catch (error: any) {
      setNotification({
        open: true,
        message: `Failed to stop training: ${error.message}`,
        severity: 'error'
      });
    }
  };

  const renderContent = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <TrainingDashboard
            status={trainingStatus}
            onPause={handlePauseTraining}
            onResume={handleResumeTraining}
            onStop={handleStopTraining}
          />
        );
      case 'config':
        return (
          <TrainingConfig
            onStartTraining={handleStartTraining}
            disabled={trainingStatus.status === 'running'}
          />
        );
      case 'history':
        return <TrainingHistory />;
      case 'models':
        return <ModelManager />;
      default:
        return null;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              R-Zero Training System
            </Typography>
            
            {trainingStatus.status === 'idle' && (
              <Button
                color="inherit"
                startIcon={<PlayArrow />}
                onClick={() => setConfigOpen(true)}
              >
                Start Training
              </Button>
            )}
            
            {trainingStatus.status === 'running' && (
              <Button
                color="inherit"
                startIcon={<Pause />}
                onClick={handlePauseTraining}
              >
                Pause
              </Button>
            )}
            
            {trainingStatus.status === 'paused' && (
              <Button
                color="inherit"
                startIcon={<PlayArrow />}
                onClick={handleResumeTraining}
              >
                Resume
              </Button>
            )}
            
            {(trainingStatus.status === 'running' || trainingStatus.status === 'paused') && (
              <Button
                color="inherit"
                startIcon={<Stop />}
                onClick={handleStopTraining}
                sx={{ ml: 1 }}
              >
                Stop
              </Button>
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Grid container spacing={2}>
                  <Grid item>
                    <Button
                      variant={currentView === 'dashboard' ? 'contained' : 'outlined'}
                      startIcon={<Timeline />}
                      onClick={() => setCurrentView('dashboard')}
                    >
                      Dashboard
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant={currentView === 'config' ? 'contained' : 'outlined'}
                      startIcon={<Settings />}
                      onClick={() => setCurrentView('config')}
                    >
                      Configuration
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant={currentView === 'history' ? 'contained' : 'outlined'}
                      onClick={() => setCurrentView('history')}
                    >
                      History
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant={currentView === 'models' ? 'contained' : 'outlined'}
                      startIcon={<Storage />}
                      onClick={() => setCurrentView('models')}
                    >
                      Models
                    </Button>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          </Grid>

          {renderContent()}
        </Container>

        {/* Training Configuration Dialog */}
        <Dialog
          open={configOpen}
          onClose={() => setConfigOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Training Configuration</DialogTitle>
          <DialogContent>
            <TrainingConfig
              onStartTraining={(config) => {
                handleStartTraining(config);
              }}
              onCancel={() => setConfigOpen(false)}
              disabled={false}
            />
          </DialogContent>
        </Dialog>

        {/* Notifications */}
        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={() => setNotification({ ...notification, open: false })}
        >
          <Alert
            onClose={() => setNotification({ ...notification, open: false })}
            severity={notification.severity}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
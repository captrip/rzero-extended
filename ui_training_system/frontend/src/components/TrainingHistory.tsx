import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Visibility,
  Refresh,
  History,
  Timeline,
  Storage
} from '@mui/icons-material';
import { format } from 'date-fns';

import { TrainingService } from '../services/TrainingService';
import { ExperimentInfo, TrainingHistory as TrainingHistoryType } from '../types/Training';

const TrainingHistory: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>([]);
  const [history, setHistory] = useState<TrainingHistoryType[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentInfo | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const trainingService = new TrainingService();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [experimentsResponse, historyResponse] = await Promise.all([
        trainingService.listExperiments(),
        trainingService.getHistory()
      ]);

      setExperiments(experimentsResponse.experiments);
      setHistory(historyResponse.history);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      return `${remainingSeconds}s`;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'completed': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const handleViewDetails = (experiment: ExperimentInfo) => {
    setSelectedExperiment(experiment);
    setDetailsOpen(true);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading training history...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Header */}
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center">
              <History sx={{ mr: 1 }} />
              <Typography variant="h5">Training History</Typography>
            </Box>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={loadData}
            >
              Refresh
            </Button>
          </Box>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
        </Grid>

        {/* Experiments List */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Experiments
              </Typography>
              
              {experiments.length === 0 ? (
                <Alert severity="info">
                  No training experiments found. Start a new training session to see results here.
                </Alert>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Experiment Name</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Start Time</TableCell>
                        <TableCell>Base Model</TableCell>
                        <TableCell>Iterations</TableCell>
                        <TableCell>Duration</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {experiments.map((experiment) => (
                        <TableRow key={experiment.experiment_id}>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                              {experiment.experiment_name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              ID: {experiment.experiment_id.slice(-8)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={experiment.status}
                              color={getStatusColor(experiment.status)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            {format(new Date(experiment.start_time * 1000), 'MMM dd, yyyy HH:mm')}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {experiment.config.base_model}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {experiment.completed_iterations || 0} / {experiment.config.num_iterations}
                          </TableCell>
                          <TableCell>
                            {experiment.total_training_time ? 
                              formatDuration(experiment.total_training_time) : 
                              'N/A'
                            }
                          </TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              startIcon={<Visibility />}
                              onClick={() => handleViewDetails(experiment)}
                            >
                              Details
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Training History */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Iteration History
              </Typography>
              
              {history.length === 0 ? (
                <Alert severity="info">
                  No training history available.
                </Alert>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Iteration</TableCell>
                        <TableCell>Questioner Model</TableCell>
                        <TableCell>Solver Model</TableCell>
                        <TableCell>Training Time</TableCell>
                        <TableCell>Timestamp</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {history.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell>
                            <Chip label={item.iteration} color="primary" size="small" />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                              {item.questioner_path.split('/').pop()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                              {item.solver_path.split('/').pop()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {formatDuration(item.training_time_seconds)}
                          </TableCell>
                          <TableCell>
                            {format(new Date(item.timestamp * 1000), 'MMM dd, HH:mm:ss')}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Experiment Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <Timeline sx={{ mr: 1 }} />
            Experiment Details
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {selectedExperiment && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  {selectedExperiment.experiment_name}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  ID: {selectedExperiment.experiment_id}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Status</Typography>
                <Chip
                  label={selectedExperiment.status}
                  color={getStatusColor(selectedExperiment.status)}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Start Time</Typography>
                <Typography variant="body2">
                  {format(new Date(selectedExperiment.start_time * 1000), 'PPpp')}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Base Model</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {selectedExperiment.config.base_model}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Iterations</Typography>
                <Typography variant="body2">
                  {selectedExperiment.completed_iterations || 0} / {selectedExperiment.config.num_iterations}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Questions per Iteration</Typography>
                <Typography variant="body2">
                  {selectedExperiment.config.questions_per_iteration}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Learning Rate</Typography>
                <Typography variant="body2">
                  {selectedExperiment.config.learning_rate.toExponential(2)}
                </Typography>
              </Grid>
              
              {selectedExperiment.total_training_time && (
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Total Training Time</Typography>
                  <Typography variant="body2">
                    {formatDuration(selectedExperiment.total_training_time)}
                  </Typography>
                </Grid>
              )}
              
              {selectedExperiment.final_models && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Final Models</Typography>
                  <Box>
                    {selectedExperiment.final_models.questioner && (
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Questioner:</strong> {selectedExperiment.final_models.questioner}
                      </Typography>
                    )}
                    {selectedExperiment.final_models.solver && (
                      <Typography variant="body2">
                        <strong>Solver:</strong> {selectedExperiment.final_models.solver}
                      </Typography>
                    )}
                  </Box>
                </Grid>
              )}
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Configuration</Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                  <pre style={{ fontSize: '0.75rem', margin: 0, overflow: 'auto' }}>
                    {JSON.stringify(selectedExperiment.config, null, 2)}
                  </pre>
                </Paper>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TrainingHistory;
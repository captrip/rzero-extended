import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button
} from '@mui/material';
import {
  Pause,
  PlayArrow,
  Stop,
  TrendingUp,
  School,
  Quiz,
  Timer
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

import { TrainingStatus } from '../types/Training';

interface TrainingDashboardProps {
  status: TrainingStatus;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}

const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  status,
  onPause,
  onResume,
  onStop
}) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'completed': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'questioner': return <Quiz />;
      case 'solver': return <School />;
      case 'training_questioner': return <Quiz />;
      case 'training_solver': return <School />;
      case 'question_generation': return <Quiz />;
      case 'evaluation': return <TrendingUp />;
      default: return <Timer />;
    }
  };

  const formatTime = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Mock data for charts (in real implementation, this would come from the backend)
  const lossData = [
    { step: 0, loss: 3.2 },
    { step: 10, loss: 2.8 },
    { step: 20, loss: 2.4 },
    { step: 30, loss: 2.1 },
    { step: 40, loss: 1.9 },
    { step: 50, loss: 1.7 },
  ];

  const progressData = [
    { iteration: 1, questioner: 100, solver: 100 },
    { iteration: 2, questioner: 100, solver: 100 },
    { iteration: 3, questioner: 100, solver: 75 },
    { iteration: 4, questioner: 0, solver: 0 },
    { iteration: 5, questioner: 0, solver: 0 },
  ];

  return (
    <Grid container spacing={3}>
      {/* Status Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <Timer sx={{ mr: 1 }} />
              <Typography variant="h6">Status</Typography>
            </Box>
            <Chip
              label={status.status.toUpperCase()}
              color={getStatusColor(status.status)}
              variant="filled"
              size="large"
            />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Experiment: {status.experiment_id?.slice(-8) || 'N/A'}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <TrendingUp sx={{ mr: 1 }} />
              <Typography variant="h6">Progress</Typography>
            </Box>
            <Typography variant="h4" color="primary">
              {status.progress_percentage.toFixed(1)}%
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={status.progress_percentage} 
              sx={{ mt: 1 }}
            />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Iteration {status.current_iteration} of {status.total_iterations}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              {getPhaseIcon(status.current_phase)}
              <Typography variant="h6" sx={{ ml: 1 }}>Current Phase</Typography>
            </Box>
            <Typography variant="h5" color="secondary">
              {status.current_phase.replace('_', ' ').toUpperCase()}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Step {status.current_step} of {status.total_steps}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Training Metrics</Typography>
            {status.loss && (
              <Typography variant="body2">
                Loss: <strong>{status.loss.toFixed(4)}</strong>
              </Typography>
            )}
            {status.learning_rate && (
              <Typography variant="body2">
                LR: <strong>{status.learning_rate.toExponential(2)}</strong>
              </Typography>
            )}
            {status.eta_seconds && (
              <Typography variant="body2">
                ETA: <strong>{formatTime(status.eta_seconds)}</strong>
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Control Panel */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Training Controls</Typography>
            <Box display="flex" gap={2}>
              {status.status === 'running' && (
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={<Pause />}
                  onClick={onPause}
                >
                  Pause Training
                </Button>
              )}
              
              {status.status === 'paused' && (
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrow />}
                  onClick={onResume}
                >
                  Resume Training
                </Button>
              )}
              
              {(status.status === 'running' || status.status === 'paused') && (
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<Stop />}
                  onClick={onStop}
                >
                  Stop Training
                </Button>
              )}
              
              {status.status === 'idle' && (
                <Typography variant="body1" color="text.secondary">
                  No training in progress. Configure and start a new training session.
                </Typography>
              )}
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Error Message */}
      {status.error_message && (
        <Grid item xs={12}>
          <Card sx={{ bgcolor: 'error.dark', color: 'error.contrastText' }}>
            <CardContent>
              <Typography variant="h6">Error</Typography>
              <Typography variant="body2">{status.error_message}</Typography>
            </CardContent>
          </Card>
        </Grid>
      )}

      {/* Training Loss Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Training Loss</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Iteration Progress Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Iteration Progress</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={progressData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="questioner"
                  stackId="1"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  name="Questioner"
                />
                <Area
                  type="monotone"
                  dataKey="solver"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  name="Solver"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Model Paths */}
      {status.model_paths && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Latest Models</Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Role</TableCell>
                      <TableCell>Path</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(status.model_paths).map(([role, path]) => (
                      <TableRow key={role}>
                        <TableCell>
                          <Chip 
                            label={role} 
                            color={role === 'questioner' ? 'secondary' : 'primary'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {path}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      )}

      {/* Phase Progress Detail */}
      {status.status === 'running' && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Current Phase Details</Typography>
              <Box sx={{ width: '100%', mb: 2 }}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">
                    {status.current_phase.replace('_', ' ').toUpperCase()}
                  </Typography>
                  <Typography variant="body2">
                    {status.current_step} / {status.total_steps}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={status.total_steps > 0 ? (status.current_step / status.total_steps) * 100 : 0}
                />
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Current Loss: {status.loss?.toFixed(4) || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Learning Rate: {status.learning_rate?.toExponential(2) || 'N/A'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );
};

export default TrainingDashboard;
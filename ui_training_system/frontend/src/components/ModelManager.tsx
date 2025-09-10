import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Storage,
  Visibility,
  Refresh,
  ExpandMore,
  School,
  Quiz,
  Download
} from '@mui/icons-material';
import { format } from 'date-fns';

import { TrainingService } from '../services/TrainingService';
import { ModelInfo } from '../types/Training';

const ModelManager: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const trainingService = new TrainingService();

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await trainingService.listModels();
      setModels(response.models);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (model: ModelInfo) => {
    setSelectedModel(model);
    setDetailsOpen(true);
  };

  const getRoleIcon = (role: string | undefined) => {
    switch (role) {
      case 'questioner': return <Quiz color="secondary" />;
      case 'solver': return <School color="primary" />;
      default: return <Storage />;
    }
  };

  const getRoleColor = (role: string | undefined) => {
    switch (role) {
      case 'questioner': return 'secondary';
      case 'solver': return 'primary';
      default: return 'default';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Group models by iteration and role for better organization
  const groupedModels = models.reduce((acc, model) => {
    const iteration = model.metadata.iteration || 0;
    const role = model.metadata.role || 'unknown';
    
    if (!acc[iteration]) {
      acc[iteration] = {};
    }
    
    acc[iteration][role] = model;
    return acc;
  }, {} as Record<number, Record<string, ModelInfo>>);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading models...</Typography>
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
              <Storage sx={{ mr: 1 }} />
              <Typography variant="h5">Model Manager</Typography>
            </Box>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={loadModels}
            >
              Refresh
            </Button>
          </Box>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Manage and view details of trained models from R-Zero training sessions.
          </Typography>
        </Grid>

        {/* Models Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Models Summary
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      {models.length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Models
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="secondary">
                      {models.filter(m => m.metadata.role === 'questioner').length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Questioner Models
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      {models.filter(m => m.metadata.role === 'solver').length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Solver Models
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Models by Iteration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Models by Training Iteration
              </Typography>
              
              {Object.keys(groupedModels).length === 0 ? (
                <Alert severity="info">
                  No trained models found. Complete a training session to see models here.
                </Alert>
              ) : (
                <Box>
                  {Object.entries(groupedModels)
                    .sort(([a], [b]) => parseInt(b) - parseInt(a)) // Sort by iteration desc
                    .map(([iteration, models]) => (
                    <Accordion key={iteration}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Box display="flex" alignItems="center" width="100%">
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            Iteration {iteration}
                          </Typography>
                          <Box sx={{ ml: 2, display: 'flex', gap: 1 }}>
                            {Object.keys(models).map(role => (
                              <Chip
                                key={role}
                                label={role}
                                color={getRoleColor(role)}
                                size="small"
                                icon={getRoleIcon(role)}
                              />
                            ))}
                          </Box>
                        </Box>
                      </AccordionSummary>
                      
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          {Object.entries(models).map(([role, model]) => (
                            <Grid item xs={12} md={6} key={role}>
                              <Card variant="outlined">
                                <CardContent>
                                  <Box display="flex" alignItems="center" mb={2}>
                                    {getRoleIcon(role)}
                                    <Typography variant="h6" sx={{ ml: 1, textTransform: 'capitalize' }}>
                                      {role} Model
                                    </Typography>
                                  </Box>
                                  
                                  <Typography variant="body2" sx={{ mb: 1, fontFamily: 'monospace' }}>
                                    {model.name}
                                  </Typography>
                                  
                                  <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                                    {model.metadata.training_steps && (
                                      <Chip label={`${model.metadata.training_steps} steps`} size="small" />
                                    )}
                                    {model.metadata.timestamp && (
                                      <Chip 
                                        label={format(new Date(model.metadata.timestamp * 1000), 'MMM dd, HH:mm')} 
                                        size="small" 
                                        variant="outlined"
                                      />
                                    )}
                                  </Box>
                                  
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    Path: {model.path}
                                  </Typography>
                                  
                                  <Button
                                    size="small"
                                    startIcon={<Visibility />}
                                    onClick={() => handleViewDetails(model)}
                                  >
                                    View Details
                                  </Button>
                                </CardContent>
                              </Card>
                            </Grid>
                          ))}
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* All Models Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                All Models
              </Typography>
              
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Role</TableCell>
                      <TableCell>Iteration</TableCell>
                      <TableCell>Training Steps</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {models
                      .sort((a, b) => (b.metadata.timestamp || 0) - (a.metadata.timestamp || 0))
                      .map((model) => (
                      <TableRow key={model.name}>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {model.name}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            {getRoleIcon(model.metadata.role)}
                            <Chip
                              label={model.metadata.role || 'unknown'}
                              color={getRoleColor(model.metadata.role)}
                              size="small"
                              sx={{ ml: 1 }}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          {model.metadata.iteration || 'N/A'}
                        </TableCell>
                        <TableCell>
                          {model.metadata.training_steps || 'N/A'}
                        </TableCell>
                        <TableCell>
                          {model.metadata.timestamp 
                            ? format(new Date(model.metadata.timestamp * 1000), 'MMM dd, yyyy HH:mm')
                            : 'N/A'
                          }
                        </TableCell>
                        <TableCell>
                          <Button
                            size="small"
                            startIcon={<Visibility />}
                            onClick={() => handleViewDetails(model)}
                          >
                            Details
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Model Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <Storage sx={{ mr: 1 }} />
            Model Details
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {selectedModel && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  {selectedModel.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Path: {selectedModel.path}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Role</Typography>
                <Box display="flex" alignItems="center">
                  {getRoleIcon(selectedModel.metadata.role)}
                  <Chip
                    label={selectedModel.metadata.role || 'unknown'}
                    color={getRoleColor(selectedModel.metadata.role)}
                    sx={{ ml: 1 }}
                  />
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Iteration</Typography>
                <Typography variant="body2">
                  {selectedModel.metadata.iteration || 'N/A'}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Training Steps</Typography>
                <Typography variant="body2">
                  {selectedModel.metadata.training_steps || 'N/A'}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>Created</Typography>
                <Typography variant="body2">
                  {selectedModel.metadata.timestamp
                    ? format(new Date(selectedModel.metadata.timestamp * 1000), 'PPpp')
                    : 'N/A'
                  }
                </Typography>
              </Grid>
              
              {selectedModel.metadata.base_model && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Base Model</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedModel.metadata.base_model}
                  </Typography>
                </Grid>
              )}
              
              {selectedModel.metadata.questions_generated && (
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Questions Generated</Typography>
                  <Typography variant="body2">
                    {selectedModel.metadata.questions_generated}
                  </Typography>
                </Grid>
              )}
              
              {selectedModel.metadata.evaluation_results && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Evaluation Results</Typography>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                    <pre style={{ fontSize: '0.75rem', margin: 0, overflow: 'auto' }}>
                      {JSON.stringify(selectedModel.metadata.evaluation_results, null, 2)}
                    </pre>
                  </Paper>
                </Grid>
              )}
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Full Metadata</Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                  <pre style={{ fontSize: '0.75rem', margin: 0, overflow: 'auto' }}>
                    {JSON.stringify(selectedModel.metadata, null, 2)}
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

export default ModelManager;
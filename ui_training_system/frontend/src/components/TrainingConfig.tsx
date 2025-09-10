import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Typography,
  FormControlLabel,
  Switch,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Cancel,
  Settings,
  Storage,
  ModelTraining
} from '@mui/icons-material';

interface TrainingConfigProps {
  onStartTraining: (config: any) => void;
  onCancel?: () => void;
  disabled: boolean;
}

const TrainingConfig: React.FC<TrainingConfigProps> = ({
  onStartTraining,
  onCancel,
  disabled
}) => {
  const [config, setConfig] = useState({
    base_model: 'meta-llama/Llama-3.2-1B',
    experiment_name: `r-zero-exp-${Date.now()}`,
    storage_path: './storage',
    huggingface_name: 'test-user',
    num_iterations: 5,
    questions_per_iteration: 1000,
    max_steps_questioner: 6,
    max_steps_solver: 20,
    learning_rate: 1e-6,
    batch_size: 4,
    save_steps: 2,
    enable_langsmith: true,
    langsmith_project: 'r-zero-training'
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (field: string, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const validateConfig = () => {
    const newErrors: Record<string, string> = {};

    if (!config.base_model.trim()) {
      newErrors.base_model = 'Base model is required';
    }

    if (!config.experiment_name.trim()) {
      newErrors.experiment_name = 'Experiment name is required';
    }

    if (!config.storage_path.trim()) {
      newErrors.storage_path = 'Storage path is required';
    }

    if (config.num_iterations < 1) {
      newErrors.num_iterations = 'Must be at least 1';
    }

    if (config.questions_per_iteration < 1) {
      newErrors.questions_per_iteration = 'Must be at least 1';
    }

    if (config.learning_rate <= 0) {
      newErrors.learning_rate = 'Must be greater than 0';
    }

    if (config.batch_size < 1) {
      newErrors.batch_size = 'Must be at least 1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateConfig()) {
      onStartTraining(config);
    }
  };

  const learningRateMarks = [
    { value: -8, label: '1e-8' },
    { value: -6, label: '1e-6' },
    { value: -4, label: '1e-4' },
    { value: -2, label: '1e-2' }
  ];

  const getLearningRateFromSlider = (value: number) => Math.pow(10, value);
  const getSliderFromLearningRate = (value: number) => Math.log10(value);

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Grid container spacing={3}>
        {/* Basic Configuration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Settings sx={{ mr: 1 }} />
                <Typography variant="h6">Basic Configuration</Typography>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Base Model"
                    value={config.base_model}
                    onChange={(e) => handleChange('base_model', e.target.value)}
                    error={!!errors.base_model}
                    helperText={errors.base_model || 'HuggingFace model name or path'}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Experiment Name"
                    value={config.experiment_name}
                    onChange={(e) => handleChange('experiment_name', e.target.value)}
                    error={!!errors.experiment_name}
                    helperText={errors.experiment_name || 'Unique name for this training run'}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Storage Path"
                    value={config.storage_path}
                    onChange={(e) => handleChange('storage_path', e.target.value)}
                    error={!!errors.storage_path}
                    helperText={errors.storage_path || 'Path to store models and data'}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="HuggingFace Username"
                    value={config.huggingface_name}
                    onChange={(e) => handleChange('huggingface_name', e.target.value)}
                    helperText="For model naming and organization"
                    disabled={disabled}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Parameters */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <ModelTraining sx={{ mr: 1 }} />
                <Typography variant="h6">Training Parameters</Typography>
              </Box>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Number of Iterations"
                    type="number"
                    value={config.num_iterations}
                    onChange={(e) => handleChange('num_iterations', parseInt(e.target.value) || 1)}
                    error={!!errors.num_iterations}
                    helperText={errors.num_iterations || 'Total R-Zero training iterations'}
                    inputProps={{ min: 1, max: 20 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Questions per Iteration"
                    type="number"
                    value={config.questions_per_iteration}
                    onChange={(e) => handleChange('questions_per_iteration', parseInt(e.target.value) || 100)}
                    error={!!errors.questions_per_iteration}
                    helperText={errors.questions_per_iteration || 'Questions generated each iteration'}
                    inputProps={{ min: 1, max: 10000 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Batch Size"
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => handleChange('batch_size', parseInt(e.target.value) || 1)}
                    error={!!errors.batch_size}
                    helperText={errors.batch_size || 'Training batch size'}
                    inputProps={{ min: 1, max: 32 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Questioner Training Steps"
                    type="number"
                    value={config.max_steps_questioner}
                    onChange={(e) => handleChange('max_steps_questioner', parseInt(e.target.value) || 1)}
                    helperText="Training steps for questioner model"
                    inputProps={{ min: 1, max: 1000 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Solver Training Steps"
                    type="number"
                    value={config.max_steps_solver}
                    onChange={(e) => handleChange('max_steps_solver', parseInt(e.target.value) || 1)}
                    helperText="Training steps for solver model"
                    inputProps={{ min: 1, max: 1000 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Learning Rate: {config.learning_rate.toExponential(2)}
                  </Typography>
                  <Slider
                    value={getSliderFromLearningRate(config.learning_rate)}
                    onChange={(_, value) => handleChange('learning_rate', getLearningRateFromSlider(value as number))}
                    min={-8}
                    max={-2}
                    step={0.1}
                    marks={learningRateMarks}
                    disabled={disabled}
                    sx={{ mt: 1 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Advanced Options */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography>Advanced Options</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Save Steps"
                    type="number"
                    value={config.save_steps}
                    onChange={(e) => handleChange('save_steps', parseInt(e.target.value) || 1)}
                    helperText="Save checkpoint every N steps"
                    inputProps={{ min: 1, max: 100 }}
                    disabled={disabled}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="LangSmith Project Name"
                    value={config.langsmith_project}
                    onChange={(e) => handleChange('langsmith_project', e.target.value)}
                    helperText="Project name for LangSmith tracking"
                    disabled={disabled || !config.enable_langsmith}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.enable_langsmith}
                        onChange={(e) => handleChange('enable_langsmith', e.target.checked)}
                        disabled={disabled}
                      />
                    }
                    label="Enable LangSmith Integration"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* LangSmith Configuration Info */}
        {config.enable_langsmith && (
          <Grid item xs={12}>
            <Alert severity="info">
              <Typography variant="body2">
                <strong>LangSmith Integration:</strong> Make sure you have set the LANGSMITH_API_KEY environment variable.
                Training metrics, experiments, and model checkpoints will be tracked in LangSmith.
              </Typography>
            </Alert>
          </Grid>
        )}

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Divider sx={{ mb: 2 }} />
          <Box display="flex" justifyContent="flex-end" gap={2}>
            {onCancel && (
              <Button
                variant="outlined"
                startIcon={<Cancel />}
                onClick={onCancel}
                disabled={disabled}
              >
                Cancel
              </Button>
            )}
            
            <Button
              type="submit"
              variant="contained"
              startIcon={<PlayArrow />}
              disabled={disabled}
              size="large"
            >
              Start Training
            </Button>
          </Box>
        </Grid>

        {/* Configuration Summary */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>Configuration Summary</Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Total Iterations: <strong>{config.num_iterations}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Questions per Iteration: <strong>{config.questions_per_iteration}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Questioner Steps: <strong>{config.max_steps_questioner}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Solver Steps: <strong>{config.max_steps_solver}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Learning Rate: <strong>{config.learning_rate.toExponential(2)}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Batch Size: <strong>{config.batch_size}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Estimated Total Steps: <strong>
                      {config.num_iterations * (config.max_steps_questioner + config.max_steps_solver)}
                    </strong>
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TrainingConfig;
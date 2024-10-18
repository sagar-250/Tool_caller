import React, { useState } from 'react';
import { Input, Button, Typography, Layout, Space, Card, Row, Col, message, Spin } from 'antd';
import axios from 'axios';

const { TextArea } = Input;
const { Title } = Typography;
const { Content } = Layout;

const OutputDisplay = ({ data }) => {
  const formatOutput = (input) => {
    console.log(input);
    
    if (Array.isArray(input)) {
      console.log("k",JSON.stringify(input, null, 2))
      return JSON.stringify(input, null, 2);
    } else if (typeof input === 'object') {
      console.log("l",JSON.stringify(input, null, 2))
      return JSON.stringify([input], null, 2);
    }
    return input;
  };

  const handleCopy = () => {
    const formattedData = formatOutput(data);
    navigator.clipboard.writeText(formattedData)
      .then(() => {
        message.success('Copied to clipboard!');
      })
      .catch(() => {
        message.error('Failed to copy!');
      });
  };

  return (
    <div>
      <pre
        style={{
          backgroundColor: '#1d1f21',
          color: '#00ff00',
          padding: '16px',
          borderRadius: '8px',
          overflowX: 'auto',
          border: '1px solid #333',
          boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
          textAlign: 'left',
        }}
      >
        <code>{formatOutput(data)}</code>
      </pre>
      <Button type="primary" onClick={handleCopy} style={{ marginTop: '16px' }}>
        Copy Output
      </Button>
    </div>
  );
};

const InputBox = ({ label, id, value, onChange, rows = 4, errorMessage }) => (
  <div style={{ marginBottom: '16px' }}>
    <label htmlFor={id} style={{ display: 'block', marginBottom: '8px', fontSize: '14px' }}>
      {label}
    </label>
    <TextArea
      id={id}
      rows={rows}
      value={value}
      onChange={onChange}
      style={{
        width: '100%',
        borderRadius: '4px',
        padding: '8px',
        fontSize: '14px',
        height: '200px',
        backgroundColor: '#1d1f21',
        color: '#00ff00',
        border: '1px solid #333',
      }}
    />
    {errorMessage && <div style={{ color: 'red', marginTop: '8px' }}>{errorMessage}</div>}
  </div>
);

const ToolCaller = () => {
  const [query, setQuery] = useState('');
  const [tools, setTools] = useState('');
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toolsError, setToolsError] = useState('');
  const [queryError, setQueryError] = useState('');

  const validateToolsInput = (toolsInput) => {
    try {
      const parsedTools = JSON.parse(toolsInput);
      if (!Array.isArray(parsedTools)) {
        return 'Input must be an array of JSON objects.';
      }
      for (const tool of parsedTools) {
        if (typeof tool !== 'object' || tool === null || Array.isArray(tool)) {
          return 'Each item in the array must be a valid JSON object.';
        }
      }
      return '';
    } catch (error) {
      return 'Invalid JSON format.';
    }
  };

  const processQuery = async () => {
    const validationError = validateToolsInput(tools);
    const queryValidationError = query.trim() === '' ? 'Query cannot be empty.' : '';

    if (validationError || queryValidationError) {
      setToolsError(validationError);
      setQueryError(queryValidationError);
      return;
    }

    setToolsError('');
    setQueryError('');
    setLoading(true); // Show loading indicator


    try {
      // Make API request to FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/toolcalling', {
        query,
        tools,
      });

     const parsedResponse = response.data;

        // Check if parsedResponse is an array of JSON strings
        const finalOutput = Array.isArray(parsedResponse)
            ? parsedResponse.map(item => JSON.parse(item)) // Parse each string into a JSON object
            : parsedResponse;

        setOutput(finalOutput); // Set the parsed API response to output
    } catch (error) {
        setToolsError('Error parsing response or invalid JSON format.');
    } finally {
        setLoading(false); // Hide loading indicator after request completes
    }
  };

  return (
    <Card style={{ padding: '24px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)', width: '100%' }}>
      <Title level={3}>Tool Caller</Title>
      <Row gutter={16}>
        <Col span={12}>
          <InputBox
            label="Query"
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            errorMessage={queryError}
          />
        </Col>
        <Col span={12}>
          <InputBox
            label="Tools (JSON format)"
            id="tools"
            value={tools}
            onChange={(e) => setTools(e.target.value)}
            errorMessage={toolsError}
          />
        </Col>
      </Row>
      <Button
        type="primary"
        onClick={processQuery}
        style={{ marginTop: '16px', marginBottom: '16px' }}
        disabled={loading} // Disable button while loading
      >
        {loading ? <Spin /> : 'Process'}
      </Button>
      {output && (
        <div>
          <Title level={4}>Output:</Title>
          <OutputDisplay data={output} />
        </div>
      )}
    </Card>
  );
};


const ToolJsonMaker = () => {
  const [toolFunctions, setToolFunctions] = useState('');
  const [output, setOutput] = useState(null);
  const [toolFunctionsError, setToolFunctionsError] = useState('');

  const generateJson = () => {
    if (toolFunctions.trim() === '') {
      setToolFunctionsError('Tool functions input cannot be empty.');
      return;
    }

    setToolFunctionsError('');

    const jsonOutput = [
      {
        tool: toolFunctions,
        functions: [
          {
            name: 'function1',
            description: 'This is function 1',
            parameters: {
              type: 'object',
              properties: {
                param1: { type: 'string' },
                param2: { type: 'number' },
              },
            },
          },
        ],
      },
    ];
    setOutput(jsonOutput);
  };

  return (
    <div className="space-y-4 bg-gradient-to-br from-blue-50 to-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-blue-800 mb-4">Tool JSON Maker</h2>
      <InputBox
        label="Tool Functions"
        id="toolFunctions"
        value={toolFunctions}
        onChange={(e) => setToolFunctions(e.target.value)}
        rows={8}
        errorMessage={toolFunctionsError}
      />
      <Button onClick={generateJson} className="bg-blue-600 hover:bg-blue-700 text-white">
        Generate JSON
      </Button>
      {output && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2 text-blue-800">Output:</h3>
          <OutputDisplay data={JSON.stringify(output, null, 2)} />
        </div>
      )}
    </div>
  );
};

const App = () => {
  const [activeSection, setActiveSection] = useState('toolCaller');

  return (
    <Layout style={{ padding: '24px', backgroundColor: '#f0f2f5', minHeight: '100vh' }}>
      <Content style={{ width: '100%', margin: '0 auto' }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: '24px' }}>
          GEAR UP
        </Title>
        <Space style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px' }}>
          <Button
            type={activeSection === 'toolCaller' ? 'primary' : 'default'}
            onClick={() => setActiveSection('toolCaller')}
          >
            Tool Caller
          </Button>
          <Button
            type={activeSection === 'toolJsonMaker' ? 'primary' : 'default'}
            onClick={() => setActiveSection('toolJsonMaker')}
          >
            Tool JSON Maker
          </Button>
        </Space>
        {activeSection === 'toolCaller' ? <ToolCaller /> : <ToolJsonMaker />}
      </Content>
    </Layout>
  );
};

export default App;

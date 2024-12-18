import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const IMVisualization = () => {
  const [imStart, setImStart] = useState(0.1);
  const [k, setK] = useState(5);
  const totalEpochs = 10;
  const imEnd = 1;

  const generateData = () => {
    return Array.from({ length: totalEpochs + 1 }, (_, epoch) => ({
      epoch,
      im: imEnd - (imEnd - imStart) * Math.exp(-k * epoch / totalEpochs)
    }));
  };

  const data = generateData();

  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      <h2 className="text-lg font-bold mb-4">IM 參數可視化工具</h2>
      <div className="mb-4">
        <label className="mr-2">im_start:</label>
        <input
          type="number"
          value={imStart}
          onChange={(e) => setImStart(Number(e.target.value))}
          min="0"
          max="1"
          step="0.1"
          className="border rounded px-2 py-1"
        />
      </div>
      <div className="mb-4">
        <label className="mr-2">k:</label>
        <input
          type="number"
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
          min="0"
          max="10"
          step="0.5"
          className="border rounded px-2 py-1"
        />
      </div>
      <LineChart width={500} height={300} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="im" stroke="#8884d8" />
      </LineChart>
    </div>
  );
};

export default IMVisualization;

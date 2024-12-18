module perceptron_t(data_out, data_0, data_1, RegDst);

input 	[4:0] data_0;
input 	[4:0] data_1;
input	RegDst;

output	[4:0] data_out;

assign  	data_out = (RegDst)?	data_1 : data_0;

endmodule
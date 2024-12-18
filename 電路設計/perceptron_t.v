module perceptron_t(w, in, out, clk, rst_n);

input 	[31:0] w;// int_bits=4, frac_bits=28
input 	[31:0] in;// single-precision floating-point
input	clk;
input	rst_n;
output reg [31:0] out;
wire	[31:0] in_s, in_r, fp_out;

assign in_s = in[30:23] <<< w;


always@(posedge clk, negedge rst_n) begin
	if(rst_n) begin
		in_s <= 0; in_r <= 0; fp_out <= 0;
	end
	else begin
		in_r <= in_s;
		FP_acc FP_acc1((in_r), (out), (fp_out));
		out <= fp_out;
	end
end

endmodule
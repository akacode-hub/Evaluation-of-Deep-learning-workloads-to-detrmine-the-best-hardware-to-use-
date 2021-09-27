// Created by Kaustubh Shivdikar
//
// (C) All Rights Reserved


#include "functions.h"
#include "debugger.h"

#include <string>

using namespace std;


Input get_inputs (int argc, char *argv[], Input input, int printer) {
		cout << "\n\n\n\n\n\n\n\n\n\n\e[1;32mProgram Start\e[0m\n";
		if (argc < 2) {
				if (printer) {
						print_inputs (argc, input);
				}
				return (input);
		}
		input.number = (long *) malloc (sizeof(long) * argc);
	
		for (int i=1; i<argc; i++) {
				input.number[i] = atol(argv[i]);
		}

		if (printer) {
				print_inputs(argc, input);
		}
		return (input); 
}

void print_inputs (int argc, Input input) {
		if (argc < 2) {
				cout << "No Inputs Provided" << endl;
		}
		for (int i=1;i<argc;i++) {
			cout << "Input[" << i << "]: " << input.number[i] << endl;
		}
}

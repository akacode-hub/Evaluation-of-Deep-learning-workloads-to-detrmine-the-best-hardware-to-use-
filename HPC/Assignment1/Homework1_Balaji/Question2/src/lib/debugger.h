// Created by Kaustubh Shivdikar
//
// (C) All Rights Reserved

#ifndef DEBUGGER_H
#define DEBUGGER_H

#define RED_START "\e[1;31m"
#define GREEN_START "\e[1;32m"
#define YELLOW_START "\e[1;33m"
#define CYAN_START "\e[1;36m"
#define UNDERLINE_START "\e[1;4m"
#define COLOR_RESET "\e[0m"

void red_start();
void green_start();
void yellow_start();
void cyan_start();
void underline_start();
void color_reset();

#endif // DEBUGGER_H

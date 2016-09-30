#ifndef LABELMETHOD_H
#define LABELMETHOD_H

// Created by Torben Trindkaer Nielsen, 1 Oct 2014
// http://www.codeproject.com/Articles/825200/An-Implementation-Of-The-Connected-Component-Label

#define CALL_LabelComponent(x,y,returnLabel) { STACK[SP] = x; STACK[SP+1] = y; STACK[SP+2] = returnLabel; SP += 3; goto START; }
#define RETURN { SP -= 3;                \
                 switch (STACK[SP+2])    \
                 {                       \
                 case 1 : goto RETURN1;  \
                 case 2 : goto RETURN2;  \
                 case 3 : goto RETURN3;  \
                 case 4 : goto RETURN4;  \
                 default: return;        \
                 }                       \
               }
#define XLAB (STACK[SP-3])
#define YLAB (STACK[SP-2])


void LabelComponent(unsigned short* STACK, unsigned short width, unsigned short height, int* input, int* output, int labelNo, unsigned short x, unsigned short y)
{
  STACK[0] = x;
  STACK[1] = y;
  STACK[2] = 0;  /* return - component is labelled */
  int SP   = 3;
  int index;

START: /* Recursive routine starts here */

  index = XLAB + width*YLAB;
  if (input [index] == 0) RETURN;   /* This pixel is not part of a component */
  if (output[index] != 0) RETURN;   /* This pixel has already been labelled  */
  output[index] = labelNo;

  if (XLAB > 0) CALL_LabelComponent(XLAB-1, YLAB, 1);   /* left  pixel */
RETURN1:

  if (XLAB < width-1) CALL_LabelComponent(XLAB+1, YLAB, 2);   /* rigth pixel */
RETURN2:

  if (YLAB > 0) CALL_LabelComponent(XLAB, YLAB-1, 3);   /* upper pixel */
RETURN3:

  if (YLAB < height-1) CALL_LabelComponent(XLAB, YLAB+1, 4);   /* lower pixel */
RETURN4:

  RETURN;
}

// Returns the labelled image as well as the number of individual components found
int LabelImage(unsigned short width, unsigned short height, int* input, int* output)
{
  unsigned short* STACK = (unsigned short*) malloc(3*sizeof(unsigned short)*(width*height + 1));

  int labelNo = 0;
  int index   = -1;
  for (unsigned short y = 0; y < height; y++)
  {
    for (unsigned short x = 0; x < width; x++)
    {
      index++;
      if (input [index] == 0) continue;   /* This pixel is not part of a component */
      if (output[index] != 0) continue;   /* This pixel has already been labelled  */
      /* New component found */
      labelNo++;
      LabelComponent(STACK, width, height, input, output, labelNo, x, y);
    }
  }

  free(STACK);

  return labelNo;
}

#endif

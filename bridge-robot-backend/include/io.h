#include <Arduino.h>


void splitStringIntoTokens(String *input, int numTokens, String *delimiters, String tokens[])
{
    int currentToken = 0;
    int currentIndex = 0;
    char currentChar = '\0';
    while (currentToken < numTokens && currentIndex < (int)(input->length()))
    {
        currentChar = input->charAt(currentIndex++);
        if (delimiters->indexOf(currentChar) != -1)
        {
            if (tokens[currentToken].length() > 0) // Only increment token if there's content
            {
                currentToken++;
            }
        }
        else
        {
            tokens[currentToken] += currentChar;
        }
    }
}
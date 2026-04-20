#ifndef __FIBSBOARD_READER_H__
#define __FIBSBOARD_READER_H__
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
enum {
	/* Token, Description */
	/* "board", first token is always "board" */
	FIBSBOARD_NAME0 = 1, /* the player's name (either you, or if you are watching someone else, that person) */
	FIBSBOARD_NAME1, /* the opponent's name */
	FIBSBOARD_MATCH_LENGTH = 3, /* match length or 9999 for unlimited matches */
	FIBSBOARD_PLAYER_GOT, /* player's points in the match so far */
	FIBSBOARD_OPPONENT_GOT, /* opponent's points in the match so far */
	FIBSBOARD_BOARD, /* 26 numbers giving the board. Positions 0 and 25 represent the bars for the players (see below). Positive numbers represent O's pieces negative numbers represent X's pieces */
	FIBSBOARD_TURN = 32, /* -1 if it's X's turn, +1 if it's O's turn 0 if the game is over */
	FIBSBOARD_DICE_0 = 33, /* 2 numbers giving the player's dice. If it's the players turn and she or he hasn't rolled, yet both numbers are 0 */
	FIBSBOARD_DICE_1 = 35, /* the opponent's dice (2 numbers) */
	FIBSBOARD_CUBE = 37, /* the number on the doubling cube */
	FIBSBOARD_MAY_DOUBLE0, /* 1 if player is allowed to double, 0 otherwise */
	FIBSBOARD_MAY_DOUBLE1, /* the same for the opponent */
	FIBSBOARD_WAS_DOUBLED, /* 1 if your opponent has just doubled, 0 otherwise */
	FIBSBOARD_COLOR, /* -1 if you are X, +1 if you are O */
	FIBSBOARD_DIRECTION, /* -1 if you play from position 24 to position 1 +1 if you play from position 1 to position 24 */
	FIBSBOARD_HOME, /* 0 or 25 depending on direction (obsolete but included anyway) */
	FIBSBOARD_FIBS_BAR, /* 25 or 0 (see home) */
	FIBSBOARD_ON_HOME0, /* number of pieces already removed from the board by player */
	FIBSBOARD_ON_HOME1, /* same for opponent */
	FIBSBOARD_ON_BAR0, /* number of player's pieces on the bar */
	FIBSBOARD_ON_BAR1, /* same for opponent */
	FIBSBOARD_CAN_MOVE, /* a number between 0 and 4. This is the number of pieces you can move. This token is valid if it's your turn and you have already rolled. */
	FIBSBOARD_FORCED_MOVE, /* don't use this token */
	FIBSBOARD_DID_CRAWFORD, /* don't use this token */
	FIBSBOARD_REDOUBLES, /* maximum number of instant redoubles in unlimited matches */
};

enum {
    FIBSBOARD_ERR_NOT_A_BOARD = -99,
    FIBSBOARD_ERR_INDEX_OUT_OF_BOUNDS = -98,
    FIBSBOARD_ERR_OUT_OF_VALUES_TO_READ = -97
};

/* Reading an integer from a fibsboard_string with some error checking. */
static inline int read_integer_from_fibsboard( const char *in, int index )
{
	if( strncmp( in, "board:", 6 ) != 0 ){ /* Check that you got a board */
		fprintf(stderr, "Trying to read fibs board integer from a string that is not a fibs board!\n"); 
		fprintf(stderr, "String is '%s'\n", in );
		return FIBSBOARD_ERR_NOT_A_BOARD;
	}
	if ( index < FIBSBOARD_MATCH_LENGTH || index > FIBSBOARD_REDOUBLES ){
		fprintf(stderr, "WARNING: (%s) Index (%d) out of range for fibs board.\n",__func__, index );
		return FIBSBOARD_ERR_INDEX_OUT_OF_BOUNDS;
	}
	char expected_endptr = (index == FIBSBOARD_REDOUBLES) ? '\0' : ':';

	char *buf = (char*) in;
	while( buf && index-- )
		buf = strchr( buf, ':' ) + 1;
	if(!buf) return FIBSBOARD_ERR_OUT_OF_VALUES_TO_READ;

	char *endptr;
	int retval = (int) strtol( buf, &endptr, 10 );
	// printf("DEBUG: At the enpointer I have '%c' \n", *endptr);
	assert( *endptr == expected_endptr ); 
	return retval;
}
#endif /* __FIBSBOARD_READER_H__ */

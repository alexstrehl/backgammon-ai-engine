#include "bg_engine.h"
#include "nn_eval.h"

#include "fibsboard_reader.h"

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/listener.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ctype.h>
#include <assert.h>

#if defined(__GNUC__)
#define UNUSED(c) c __attribute__((__unused__))
#else
#define UNUSED(c)
#endif

#define DEFAULT_PORT 9876

static char **whitelist_ips = NULL;  /* I don't like globals but I'm doing an exception */

static void read_board_from_message( BoardState *board, const char *s )
{
	/*
	Token			Description
	"board"			first token is always "board"
	name			the player's name (either you, or if you are watching someone else, that person)
	name			the opponent's name
	match length	match length or 9999 for unlimited matches
	player got		player's points in the match so far
	opponent got	opponent's points in the match so far
	board			26 numbers giving the board. Positions 0 and 25 represent the bars for
					the players (see below). Positive numbers represent O's pieces negative
					numbers represent X's pieces
	turn			-1 if it's X's turn, +1 if it's O's turn 0 if the game is over
	dice			2 numbers giving the player's dice. If it's the players turn and she or he
					hasn't rolled, yet both numbers are 0
	dice			the opponent's dice (2 numbers)
	cube			the number on the doubling cube
	may double		1 if player is allowed to double, 0 otherwise
	may double		the same for the opponent
	was doubled		1 if your opponent has just doubled, 0 otherwise
	color			-1 if you are X, +1 if you are O
	direction		-1 if you play from position 24 to position 1 +1 if you play from position 1 to position 24
	home			0 or 25 depending on direction (obsolete but included anyway)
	bar				25 or 0 (see home)
	on home			number of pieces already removed from the board by player
	on home			same for opponent
	on bar			number of player's pieces on the bar
	on bar			same for opponent
	can move		a number between 0 and 4. This is the number of pieces you can move. This
					token is valid if it's your turn and you have already rolled.
	forced move		don't use this token
	did crawford	don't use this token
	redoubles		maximum number of instant redoubles in unlimited matches
	*/

	memset( board, 0, sizeof( BoardState ));
	int offset;
	const char *pch = strchr(s,':') + 1; /* the first : after 'board'; */

	/* At the current state these values are just ignored anyway. */ 
	char name[2][256];
	for( int i = 0; i < 2 ; i++, pch += offset )
		if( 1 != sscanf( pch, "%[^:]:%n", name[i], &offset )) {
			printf("Cannot read board a character '%c'.\n", *pch); 
			return;
		}

	/* Note that these macros declares new variables. C99 or later is therefor mandatory */
#define read_int(vname) \
	int vname; \
	if( !sscanf( pch, "%d:%n", &vname, &offset )){ \
		printf("Cannot read board a character '%c'.\n", *pch); \
		return; \
	} else { pch += offset; }
	
#define read_int_array(vname, size) \
	int vname[size]; \
	for( int i = 0; i < size ; i++, pch += offset ) \
		if( !sscanf( pch, "%d:%n", &vname[i], &offset )){ \
			printf("Cannot read board a character '%c'.\n", *pch); \
			return; \
		}

	read_int      ( matchlength );
	read_int_array( score, 2 );
	read_int_array( board_array, 26 );
	read_int      ( turn );
	read_int_array( dice, 4 );
	read_int      ( cube );
	read_int_array( may_double, 2 );
	read_int      ( was_double );
	read_int      ( color );       /* Always  1 for GNU Backgammon produced boards */
	read_int      ( direction );   /* Always -1 for GNU Backgammon produced boards */

	read_int      ( home );        /* Always  0 for GNU Backgammon produced boards */
	read_int      ( bar );         /* Always 25 for GNU Backgammon produced boards */

	read_int_array( on_home, 2 );
	read_int_array( on_bar, 2 );   /* Always 0:0 for GNU Backgammon produced boards */

#undef read_int_array
#undef read_int
#if 0
#define debug_fibs_print(vname) \
	printf( "DEBUG: %12s: %2d\n", #vname, vname);
#else
#define debug_fibs_print(vname) {};
#endif
	debug_fibs_print( matchlength );
	debug_fibs_print( score[0] );
	debug_fibs_print( score[1] );
	debug_fibs_print( turn );
	debug_fibs_print( dice[0] );
	debug_fibs_print( dice[1] );
	debug_fibs_print( dice[2] );
	debug_fibs_print( dice[3] );
	debug_fibs_print( cube );
	debug_fibs_print( may_double[0] );
	debug_fibs_print( may_double[1] );
	debug_fibs_print( was_double );
	debug_fibs_print( color );
	debug_fibs_print( direction );

	debug_fibs_print( home );
	debug_fibs_print( bar );

	debug_fibs_print( on_home[0] );
	debug_fibs_print( on_home[1] );
	debug_fibs_print( on_bar[0] );
	debug_fibs_print( on_bar[1] );
#undef debug_fibs_print

	board->bar[BLACK] = -board_array[0];
	board->bar[WHITE] = board_array[25];
	for ( int i = 0  ; i < 24 ; i++ )
		board->points[i] = board_array[i+1];

	/* I believe that if direction is 1, I have to swap the bar-points?
	   Really? I'll test this and if true, I will find a more elegant way.
	   However, that said - as GNU Backgammon server direction will always be -1, so
	   this thing will never be executed if connected as external player with gnubg.
	 */
	if ( direction == 1 ){
		int tmp = board->bar[WHITE];
		board->bar[WHITE] = board->bar[BLACK];
		board->bar[BLACK] = tmp;
	}

	board->off[WHITE] = board->off[BLACK] = NUM_CHECKERS;
	for ( int i = 0; i < 24; i++ ){
		if( !board->points[i] ) continue;
		if( board->points[i] > 0 ) board->off[WHITE] -= board->points[i];
		if( board->points[i] < 0 ) board->off[BLACK] += board->points[i];
	}

	board->off[WHITE] -= board->bar[WHITE];
	board->off[BLACK] -= board->bar[BLACK];

	board->turn = turn == color ? WHITE : BLACK;
}

#define NUM_FEATURES_MAXIMUM 200
static int choose_play(const NNModel *model, const Play *plays, int num_plays) {
	assert( model->input_size <= NUM_FEATURES_MAXIMUM);
	float features[NUM_FEATURES_MAXIMUM] = {0};
	float best_val = 20.0f;
	int best_idx = 0;

	for (int i = 0; i < num_plays; i++) {
		BoardState s = plays[i].resulting_state;
		// board_switch_turn(&s);  /* Using v2 of get_legal_plays() which flipps the turn automagically */
		encode_state(&s, features);
		/* FIXME: Add the cube position features (intentionally omitted as I think the nn will change here) */		
		float val = nn_forward(model, features);
		if (val < best_val) {
			best_val = val;
			best_idx = i;
		}
	}
	return best_idx;
}

static char *movestring_from_play( char *buffer, Play play )
{
	char *ptr = buffer;
	for( int i = 0; i < play.num_moves; i++ ){
		if( play.moves[i].src == BAR_SENTINEL )
			ptr += sprintf( ptr, "bar %d ", 24-play.moves[i].dst );
		else if ( play.moves[i].dst == OFF_SENTINEL )
			ptr += sprintf( ptr, "%d off ", 24-play.moves[i].src );
		else
			ptr += sprintf( ptr, "%d %d ", 24 - (play.moves[i].src), 24-play.moves[i].dst );
	}
	return buffer;
}

static int take_or_drop(const NNModel *model, const BoardState *s )
{
	assert( model->input_size <= NUM_FEATURES_MAXIMUM);
	float features[NUM_FEATURES_MAXIMUM] = {0};
	encode_state( s, features);
	features[199] = 1.0f; /* FIXME: Bad code. I code this because I know the nn structure */		
	float val = nn_forward(model, features);
	return (val > 0.5f) ? 0 : 1;
}

static int double_or_roll(const NNModel *model, const BoardState *s, int cubeowner )
{
	assert( cubeowner == 0 || cubeowner == 1 );
	assert( model->input_size <= NUM_FEATURES_MAXIMUM);
	float features[NUM_FEATURES_MAXIMUM] = {0};
	encode_state( s, features);
	features[199] = 1.0f; /* FIXME: Bad code. I code this because I know the nn structure */		
	float double_take_val = 2 * nn_forward(model, features);
	features[199] = 0.0f;
	features[197 + cubeowner] = 1.0f;
	float no_double_val = nn_forward(model, features);

	return (no_double_val > double_take_val) ? 0 : 1;
}

#define MESSAGE_BUFFER_SIZE 1024
void read_cb(struct bufferevent *bev, void *ptr) {
	char buffer[MESSAGE_BUFFER_SIZE];
	int n;
	NNModel *nn = (NNModel*) ptr;

	// Read data from the client
	while ((n = bufferevent_read(bev, buffer, sizeof(buffer))) > 0) {
		if( strncmp( buffer, "GET", 3 ) == 0 || strncmp( buffer, "POST", 4 ) == 0){ 
			evbuffer_add_printf(bufferevent_get_output(bev), "HTTP/1.1 200 OK\n\nThis is not a web server - connect with GNU Backgammon external player. Server closing.\n" );
			break;
        }
		assert( n < MESSAGE_BUFFER_SIZE );	
		buffer[n] = '\0'; // Null-terminate the buffer (just to be sure!)
		printf("Got: %s", buffer);

		/* Build a BoardState from the string */
		BoardState bs;
		read_board_from_message( &bs, buffer );
		int dice[2];
		for ( int i = 0; i < 2 ; i++ )
			dice[i] = read_integer_from_fibsboard( buffer, FIBSBOARD_DICE_0 + i );
		if( dice[0] == 0 && dice[1] == 0 ){
			/* No dice */
			bool did_double = (bool) read_integer_from_fibsboard( buffer, FIBSBOARD_WAS_DOUBLED );
			if ( did_double ){
				evbuffer_add_printf(bufferevent_get_output(bev), "%s\n", take_or_drop(nn, &bs) ? "take" : "drop" );
			} else {
				int cubeowner = read_integer_from_fibsboard( buffer, FIBSBOARD_CUBE ) != 1 ? 1 : 0; /* FIXME This is weak code */
				evbuffer_add_printf(bufferevent_get_output(bev), "%s\n", double_or_roll(nn, &bs, cubeowner) ? "double" : "roll" );
			}
		} else {
			printf("I should find the best move with: %d%d\n", dice[0], dice[1]);
			Play plays[MAX_PLAYS];
			int num_plays = get_legal_plays(&bs, dice[0], dice[1], plays, MAX_PLAYS);

			if (num_plays > 0) {
				int best = choose_play(nn, plays, num_plays);
				/* bar 24 bar 24 bar 24 bar 24 */
				/* 123456789012345678901234567890 */
				/* I think 30 is enough for the buffer size - but hey, I can make it 64 if it makes you happy, Claude */
				char movebuffer[64] = {0};
				movestring_from_play( movebuffer, plays[best] );
				printf("I think the best move is: %s\n", movebuffer );
				evbuffer_add_printf(bufferevent_get_output(bev), "%s\n", movebuffer );
			} else {
				evbuffer_add_printf(bufferevent_get_output(bev), "\n" );
			}
		}
	}
}

void event_cb(struct bufferevent *bev, short events, void *nn) {
	if (events & BEV_EVENT_ERROR) {
		perror("Error on the connection");
	}
	if (events & (BEV_EVENT_EOF | BEV_EVENT_ERROR)) {
		printf("Connection closed\n");
		bufferevent_free(bev); // Free the bufferevent
	}

	/* Free the neuralnet? */
	/* FIXME: Oh! If there's several connections - the server will close down if one of the connection closes! */
	/* We may solve this by counting the number of connections and release the server only when the count reaches 0. */
	/* Well - maybe the server should just wait for a kill signal to clean up and exit? */
	/* Whatever - it might be an idea to count the number of connections and set a limit */
	if( nn ) {
		nn_free( nn );
		free( nn );
	}
	if ( whitelist_ips )
		free( whitelist_ips );

	exit( EXIT_SUCCESS );
}

#define foreach_str(iter, ...) \
	for( char **iter = (char*[]){__VA_ARGS__, NULL}; *iter; iter++ )
#define streq(a,b) (strcmp((a),(b)) == 0)

void accept_cb(struct evconnlistener *listener, evutil_socket_t fd,
		UNUSED(struct sockaddr *addr), UNUSED(int socklen), void *nn) {

	// Create a new bufferevent for the incoming connection
	struct event_base *base = evconnlistener_get_base(listener);
	struct bufferevent *bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);

	if (!bev) {
		fprintf(stderr, "Error creating bufferevent\n");
		event_base_loopbreak(base);
		return;
	}
	/* Note - these stuff is POSIX and not ANSI C - use -std=gnu99 to your compiler if these lines fail */
#define IP_STR_BUFSIZE 40
	char ip_str[IP_STR_BUFSIZE + 1] = {0};
	int result = getnameinfo(addr, socklen, ip_str, IP_STR_BUFSIZE, NULL, 0, NI_NUMERICHOST);
#undef IP_STR_BUFSIZE
	if( result == 0 ) 
		printf("Got a new connection from '%s'\n", ip_str);
	else
		printf("Got a new connection -- failed to find IP: %s\n", gai_strerror(result));

	bool whitelisted = false;
	for( char **p = whitelist_ips; *p; p++ ){
		if( streq( *p, ip_str )){
			whitelisted = true;
			break;
		}
	}
	if( !whitelisted || result != 0 ){
		printf("None whitelisted connection. Closing.... \n");
		bufferevent_free( bev );
		return;
	}

	// Set callbacks for reading and events
	bufferevent_setcb(bev, read_cb, NULL, event_cb, nn);
	bufferevent_enable(bev, EV_READ | EV_WRITE);
}
#if 1
static inline char *strstrip(char *str)
{
	size_t size;
	char *end, *start;

	if( !str ) return NULL;

	size = strlen(str);

	if (!size) return str;

	end = str + size - 1;
	while (end >= str && isspace(*end))
		end--;
	*(end + 1) = '\0';

	start = str;
	while ( isspace(*start ) )
		start++;

	return (char*) memmove( str, start, end - start + 2);
}
#endif 

/* Parses a line - expects a ':' somewhere in the input string */
static void parse_line( char *line, char **kw, char **v )
{
	char *split = strchr( line, ':' );
	if (!split) {
		/* No ':' found, setting NULL and returning */
		*kw = NULL;
		*v  = NULL;
		return;
	}

	*split = '\0';
	*kw = line;
	*v = split + 1;
	strstrip( *kw );
	strstrip( *v );
	return;
}

static char **whitelist_split( char *in, char delimiter ) 
{ 
	unsigned int i = 0, count = 0;
	char *pstr; 
	if ( *in == '\n' || *in == '\0' ){ /* Blank */
		char **ret = malloc( 2 * sizeof( char *));
		if ( !ret ) return NULL;
		ret[0] = "127.0.0.1";
		ret[1] = NULL;
		return ret;
	}

	for ( i = 0; in[i] != '\0' ; i++ )
		count += in[i] == delimiter ? 1 : 0;

	char **strv = (char **) malloc( (3 + i) * sizeof ( char* )); 
	/* Why plus 3? 
	 * - there is one more item in the list than delimiting characters.
	 * - then there is a terminating NULL.
	 * - and then we want to add localhost 127.0.0.1 for convienience.
	 */
	if ( !strv ) return NULL;	 
	strv[ 0 ] = "127.0.0.1";
	strv[ 1 ] = in; 

	for ( pstr = in, i = 2; *pstr != '\0'; pstr++ ){ 
		if( *pstr == delimiter ) { 
			*pstr = '\0'; 
			strv[ i++ ] = pstr + 1;
		}
	}
	strv[ i ] = NULL; 

	return strv; 
}

static bool is_blank_line( const char *l )
{
	for( int i = 0; l[i] != '\0'; i++)
		if ( l[i] != ' ' && l[i] != '\t' && l[i] != '\r' && l[i] != '\n' )
			return false;
	return true;
}

void usage( const char *appname )
{
	printf("Usage: %s [options]\n", appname );
	printf("\nOptions:\n\n");
	printf("  --modelfile=<filename>     Model file for the AI.\n");
	printf("  --mode=[dmp|money]         (not in use)\n");
	printf("  --port=<integer>           Port number for the server (default %d)\n", DEFAULT_PORT);
	printf("  --whitelist=<ip,ip,..,ip>  Comma separated list of whitelisted IPs. (127.0.0.1 is always added.)\n" );
	printf("  --help                     Show this text.\n\n" );
}


int main(int argc, char *argv[])
{
	printf("This is a backgammon player server for 'external player' in GNU Backgammon.\n"
			"Build date: " __DATE__ " ( " __TIME__ " )\n");


	char modelfile[64] = {};
	char mode[8] = "money";   /* Should be 'money' or 'dmp' at this stage. */
	int  port = DEFAULT_PORT;
	char whitelist[256] = "";

	char *config_file = NULL;
	foreach_str( cf, ".bgserver.conf", "~/.bgserver.conf") {
		if( access( *cf, R_OK ) == 0 ){
			config_file = *cf;
			break;
		}
	}
#define PARSE_CONFIG_INT(kw, arg, val) \
	if( strncmp( arg, #kw, strlen(#kw) ) == 0 ){ \
		kw = strtol( val, NULL, 10 ); \
		continue; \
	}
#define PARSE_CONFIG_STR(kw, arg, val) \
	if( strncmp( arg, #kw, strlen(#kw) ) == 0 ){ \
		snprintf( kw, strlen(kw), "%s", val); \
		continue; \
	}

	if ( config_file ){
		FILE *fp = fopen( config_file, "r");
		if( fp ){
			char line[256];
			int line_number = 0;
			while( fgets( line, sizeof(line), fp ) != NULL ){
				line_number++;
				if( line[0] == '#' ) continue;
				if( is_blank_line(line) ) continue;
				char *keyword, *value;
				parse_line( line, &keyword, &value );
				if ( (keyword == NULL ) && (value == NULL)){
					fprintf(stderr, "Cannot parse line: '%d' in config file '%s'.\n", line_number, config_file);
					fprintf(stderr, "Keyword and value should be separated by ':'.\n");
					continue;
				}
				PARSE_CONFIG_STR(modelfile, keyword, value);
				PARSE_CONFIG_STR(mode     , keyword, value);
				PARSE_CONFIG_INT(port     , keyword, value);
				PARSE_CONFIG_STR(whitelist, keyword, value);
				fprintf(stderr, "Cannot parse line: '%d' in file '%s' (ignoring)\n", line_number, config_file );
				fprintf(stderr, "Keyword '%s'\n", keyword );
			}
			fclose( fp );
		} else {
			perror( config_file );
		}
	} else {
		printf("No config file found.\n");
	}

#define OPTARG_READ_INT(optarg, arg) \
	if( strncmp( arg, "--" #optarg, strlen("--" #optarg) ) == 0 ){ \
		char *endptr = NULL;                                       \
		char *where_is_equal = strchr( arg, '=' );                 \
		if( where_is_equal ) optarg = strtol( where_is_equal + 1, &endptr, 10 ); \
		continue; \
	}
#define OPTARG_READ_STRING(optarg, arg) \
	if( strncmp( arg, "--" #optarg, strlen("--" #optarg) ) == 0 ){ \
		char *where_is_equal = strchr( arg, '=' );                 \
		if( where_is_equal ) snprintf( optarg, sizeof(optarg), "%s", where_is_equal + 1 ); \
		continue; \
	}

	for ( int i = 1 ; i < argc ; i++ ){
		if( strncmp( argv[i], "--help", 6 ) == 0 || strncmp(argv[i], "-h", 2) == 0 ) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		}
		OPTARG_READ_INT(port, argv[i]);
		OPTARG_READ_STRING(modelfile, argv[i]);
		OPTARG_READ_STRING(mode,      argv[i]);
		OPTARG_READ_STRING(whitelist, argv[i]);
		fprintf(stderr, "WARNING: Unrecognized argument: %s\n", argv[i] );
	}

	whitelist_ips = whitelist_split( whitelist, ',' );
	for( int i = 0; whitelist_ips[i] != NULL; i++){
		printf("whitelisted ip %2d: %s\n", i, whitelist_ips[i]);
	}

	NNModel *nn = malloc( sizeof( NNModel ));
	if (!nn ) {
		fprintf(stderr, "Can't allocate memory for neural network model.\n" ); 
		free( whitelist_ips );
		return 1;
	}
	if( nn_load( nn, modelfile )){
		fprintf(stderr, "Can't create neural network from '%s'\n", modelfile ); 
		free( whitelist_ips );
		return 1;
	}

	// Initialize the event base
	struct event_base *base = event_base_new();
	if (!base) {
		fprintf(stderr, "Error creating event base\n");
		nn_free( nn );
		free( nn );
		free( whitelist_ips );
		return 1;
	}

	if( port <= 1024 || port > 65535){
		fprintf(stderr, "Specified port number (%d) is out of range. (1025-65535). Using default (%d)\n",
				port, DEFAULT_PORT);
		port = DEFAULT_PORT;
	}

	// Set up the sockaddr_in structure for the server
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = htonl(0); // Bind to all interfaces
	sin.sin_port = htons(port);     // Port to listen on

	// Create a listener for incoming connections
	struct evconnlistener *listener = evconnlistener_new_bind(base, accept_cb, nn,
			LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, -1,
			(struct sockaddr *)&sin, sizeof(sin));
	if (!listener) {
		perror("Error creating listener");
		event_base_free(base);
		nn_free( nn );
		free( nn );
		free( whitelist_ips );
		return 1;
	}

	printf("Listener started at port %d\n", port);

	// Start the event loop
	event_base_dispatch(base);

	// Clean up
	evconnlistener_free(listener);
	event_base_free(base);
	nn_free( nn );
	free( nn );
	free( whitelist_ips );

	return 0;
}


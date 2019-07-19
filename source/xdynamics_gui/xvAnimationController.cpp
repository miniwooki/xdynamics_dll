#include "xvAnimationController.h"
/*#include "Object.h"*/

int xvAnimationController::play_mode = 0;
unsigned int xvAnimationController::current_frame = 0;
unsigned int xvAnimationController::buffer_count = 0;
bool xvAnimationController::is_play = false;
float *xvAnimationController::times = 0;// new float[1000];
bool xvAnimationController::real_time = false;

xvAnimationController::xvAnimationController()
{

}

xvAnimationController::~xvAnimationController()
{
	
}

void xvAnimationController::releaseTimeMemory()
{
	if (times) delete[] times;
}

bool xvAnimationController::is_end_frame()
{
	if (is_play){
		/*current_frame++;*/
		if (current_frame > buffer_count - 1){
			current_frame = buffer_count;
			return true;
		}
	}

	return false;
}

void xvAnimationController::move2previous2x()
{
	current_frame ? (--current_frame ? --current_frame : current_frame = 0) : current_frame = 0;
}

void xvAnimationController::move2previous1x()
{
	current_frame ? --current_frame : current_frame = 0;
}

void xvAnimationController::on_play()
{
	is_play = true;
}

bool xvAnimationController::Play()
{
	return is_play;
}

void xvAnimationController::off_play()
{
	is_play = false;
}

void xvAnimationController::move2forward1x()
{
	current_frame == buffer_count ? current_frame = current_frame : ++current_frame;
}

void xvAnimationController::move2forward2x()
{
	current_frame == buffer_count ? current_frame = current_frame : (++current_frame == buffer_count - 1 ? current_frame = current_frame : ++current_frame);
}

void xvAnimationController::update_frame()
{
	switch (play_mode)
	{
	case 1:	move2previous2x(); break;
	case 3: move2previous1x(); break;
	case 5: move2forward1x(); break;
	case 7: move2forward2x(); break;
	}
}
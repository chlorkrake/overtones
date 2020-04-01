# -*- coding: utf-8 -*-
from constants import *
import scipy.integrate

from manimlib.imports import *

USE_ALMOST_FOURIER_BY_DEFAULT = True
NUM_SAMPLES_FOR_FFT = 1000
DEFAULT_COMPLEX_TO_REAL_FUNC = lambda z : z.real

def get_fourier_transform(
    func, t_min, t_max, 
    complex_to_real_func = DEFAULT_COMPLEX_TO_REAL_FUNC,
    use_almost_fourier = USE_ALMOST_FOURIER_BY_DEFAULT,
    **kwargs ##Just eats these
    ):
    scalar = 1./(t_max - t_min) if use_almost_fourier else 1.0
    def fourier_transform(f):
        z = scalar*scipy.integrate.quad(
            lambda t : func(t)*np.exp(complex(0, -TAU*f*t)),
            t_min, t_max
        )[0]
        return complex_to_real_func(z)
    return fourier_transform


class Block(Rectangle):
    CONFIG = {
        "mass": 1,
        "velocity": 0,
        "width": 0.2,
        "height": 6.0,
        "fill_opacity": 1,
        "stroke_width": 1,
        "stroke_color": WHITE,
        "fill_color": None,
        
    
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Rectangle.__init__(self, side_length=self.width, **kwargs)

class AirMolecule(Circle):
    CONFIG = {
        "velocity": 0,
        "radius": 0.08,
        "fill_opacity": 1,
        "stroke_width": 0.1,
        "stroke_color": BLUE,
        "fill_color": BLUE,
        
    
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Circle.__init__(self, **kwargs)

class SlidingBlocks(VGroup):
    CONFIG = {
        "block_config": {
            "distance": 1,
            "mass": 1,
            "velocity": 0,
        },
        "membrane":-4.5,
        "floor_pos":-3,
        "amplitude": 0.4, #0.4
        "frequency":3, #1
        "timer":0,
    }

    def __init__(self, scene, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.scene = scene
        self.block = self.get_block(**self.block_config)
        self.point_field = self.get_points()
        
        self.add(
            self.block,
        )
        self.add_updater(self.__class__.update_positions)



    def get_block(self, distance, **kwargs):
        block = Block(**kwargs)
        block.move_to(
            self.floor_pos * DOWN +
            (self.membrane + distance) * LEFT,
            DL,
        )
        return block

    def get_points(self,**kwargs):
        points = [x*RIGHT+y*UP
            for x in np.arange(-4.1,8,0.25) #-4
            for y in np.arange(-3,3,0.25) 
            ]
        point_field=[]
        for point in points:
            circle=AirMolecule().shift(point)
            circle.x = point[0]
            circle.y = point[1]
            point_field.append(circle)
            self.add(circle) 
        return point_field
            

  
    def update_positions(self, dt):
        block= self.block
        floor_y = self.floor_pos
        #point_field = self.point_fieldmju765
        self.timer +=dt
        ps_block= self.amplitude* np.sin(self.frequency*self.timer )
        block.move_to(
                (self.membrane+ ps_block) * RIGHT +
                floor_y * UP,      
                DL,
            )
            
        for point in self.point_field:
            ps_point = self.amplitude*np.sin(self.frequency * (self.timer - point.x)) #cos
            point.move_to(
                (point.x+ ps_point) * RIGHT +
                point.y * UP,      
                DL,
            )




class MovingString(VGroup):
    CONFIG = {
        "amplitude": 0.5,
        "frequency":0.25,
        "timer":0,
    }

    def __init__(self, scene, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.scene = scene
        self.point_field = self.get_points()
        self.add_updater(self.__class__.update_positions)



    def get_block(self, distance, **kwargs):
        block = Block(**kwargs)
        block.move_to(
            self.floor_pos * DOWN +
            (self.membrane + distance) * LEFT,
            DL,
        )
        return block

    def get_points(self,**kwargs):
        points = [x*RIGHT
            for x in np.arange(-2*np.pi,2*np.pi,np.pi/20)
            for y in np.arange(-3,3,0.3)
            ]
        point_field=[]
        for point in points:
            circle=AirMolecule().shift(point)
            circle.x = point[0]
            circle.y = 0
            point_field.append(circle)
            self.add(circle) 
        return point_field
            

  
    def update_positions_working(self, dt):
        self.timer +=dt
        
        for point in self.point_field:

            ps_point = (self.amplitude*np.sin(self.timer*5))*np.cos(self.frequency*point.x) 
            #ps_point += ((self.amplitude*np.sin(self.timer*5))*np.sin(self.frequency*4*point.x))/4
            ps_point +=  (self.amplitude*np.sin(self.timer*5)*np.sin(self.frequency*6*point.x))/4
            ps_point +=  (self.amplitude*np.sin(self.timer*5)*np.sin(self.frequency*8*point.x))/4
            ps_point +=  (self.amplitude*np.sin(self.timer*5)*np.sin(self.frequency*10*point.x))/4
            point.move_to(
                point.x*RIGHT+
                (ps_point) * UP,   
                DL   
            )

        


class WhatIsATone(Scene):
    CONFIG = {
        "A_frequency" : 2.1,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 0,
    }

    def construct(self):
        self.question=TextMobject("Was ist ein Ton?")
        self.question.scale_in_place(1.3)
        self.play(FadeIn(self.question))
        self.headline = deepcopy(self.question)
        self.headline.shift(3*UP)
        self.play(Transform(self.question,self.headline))
        self.show_sine()

    def show_sine(self):
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 12,
            number_line_config = {"include_tip" : False},
        )
        axes.stretch_to_fit_height(2)
        axes.to_corner(2*LEFT)
        #axes.shift(UP)
       
        frequency = self.A_frequency
        graph = self.get_wave_graph(frequency, axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)

        
        
        frequency = self.A_frequency
        graph = self.get_wave_graph(frequency, axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)
        pressure = TextMobject("Luftdruck")
        time = TextMobject("Zeit")
        for label in pressure, time:
            label.scale_in_place(0.8)
        pressure.next_to(axes.y_axis, UP)
        pressure.to_edge(LEFT, buff = MED_SMALL_BUFF)
        time.next_to(axes.x_axis.get_right(), DOWN+LEFT)
        axes.labels = VGroup(pressure, time)

        n = 10
        brace = Brace(Line(
            axes.coords_to_point(n/frequency, func(n/frequency)),
            axes.coords_to_point((n+1)/frequency, func((n+1)/frequency)),
        ), UP)
        words = brace.get_text("440 pro Sekunde", buff = SMALL_BUFF)
        words.scale(0.8, about_point = words.get_bottom())

        self.play(
            ShowCreation(graph, run_time = 6, rate_func=linear),
            #ShowCreation(equilibrium_line),
        )
        self.wait(3)
        self.play(
            FadeIn(pressure),
            ShowCreation(axes.y_axis)
        )
        self.play(
            Write(time),
            ShowCreation(axes.x_axis)
        )
        self.wait(10)

    def get_wave_graph(self, frequency, axes):
        tail_len = 3.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.7*np.cos(2*np.pi*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph
        
 
class AirPressureFast(Scene):
    CONFIG = {
        "sliding_blocks_config": {
            "block_config": {
                "mass": 1e0,
                "velocity": -2,
            }
        },
        "wait_time": 10,
        "frequency" : 1,
        "amplitude":2,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
        "plane_kwargs" : {},
    }
    def setup(self):
        plane = NumberPlane(**self.plane_kwargs)
        self.track_time()
        self.add_blocks_and_points()
       
        

    def add_blocks_and_points(self):
        self.blocks = SlidingBlocks(self, **self.sliding_blocks_config)
        self.add(self.blocks)

    def track_time(self):
        time_tracker = ValueTracker()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(time_tracker)
        self.get_time = time_tracker.get_value



    def construct(self):
        self.wait(10)
        


    def get_wave_graph(self, axes):
        tail_len = 1.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = self.amplitude*np.cos(2*np.pi*self.frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*self.frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph
                   

class FrequencyPlot(Scene):
    CONFIG = {
        "A_frequency" : 2.1,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
    }
    def construct(self):
        self.play_A440()
        self.play_lower_pitch()
        self.wait(10)




    def play_A440(self):
        self.axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 12,
            number_line_config = {"include_tip" : False},
        )
        axes = self.axes
        A_label = TextMobject("A440")
        A_label.set_color(self.A_color)
        #A_label.to_edge(DOWN)
        #A_label.shif(UP)
        self.set_variables_as_attrs(A_label)
    
        
        axes.stretch_to_fit_height(2)
        axes.to_corner(UP+LEFT)
        axes.shift(1.5*DOWN)
        eh = self.equilibrium_height
        equilibrium_line = DashedLine(
            axes.coords_to_point(0, eh),
            axes.coords_to_point(axes.x_max, eh),
            stroke_width = 2,
            stroke_color = LIGHT_GREY
        )

        frequency = self.A_frequency
        graph = self.get_wave_graph(frequency, axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)
        pressure = TextMobject("Luftdruck")
        time = TextMobject("Zeit")
        for label in pressure, time:
            label.scale_in_place(0.8)
        pressure.next_to(axes.y_axis, UP)
        pressure.to_edge(LEFT, buff = MED_SMALL_BUFF)
        time.next_to(axes.x_axis.get_right(), DOWN+LEFT)
        axes.labels = VGroup(pressure, time)

        n = 10
        brace = Brace(Line(
            axes.coords_to_point(n/frequency, func(n/frequency)),
            axes.coords_to_point((n+1)/frequency, func((n+1)/frequency)),
        ), UP)
        words = brace.get_text("440 pro Sekunde", buff = SMALL_BUFF)
        words.scale(0.8, about_point = words.get_bottom())

        self.play(
            FadeIn(pressure),
            ShowCreation(axes.y_axis)
        )
        self.play(
            Write(time),
            ShowCreation(axes.x_axis)
        )
        self.play(
            FadeIn(A_label),
            
        )
        self.play(
            ShowCreation(graph, run_time = 4, rate_func=linear),
            ShowCreation(equilibrium_line),
        )
        axes.add(equilibrium_line)
        self.play(
            GrowFromCenter(brace),
            Write(words)
        )
        self.wait(2)
        graph.save_state()
        self.play(
            FadeOut(brace),
            FadeOut(words),
        )

        self.set_variables_as_attrs(axes, A_graph = graph)




    def play_lower_pitch(self):
        axes = self.axes
        D_axes= deepcopy(axes)
        D_axes.shift(3*DOWN)
        frequency = self.A_frequency*(2.0/3.0)
        graph = self.get_wave_graph(frequency, D_axes)
        graph.set_color(self.D_color)

        D_label = TextMobject("D294")
        D_label.set_color(self.D_color)
        D_label.shift(3*DOWN)

        pressure = TextMobject("Luftdruck")
        time = TextMobject("Zeit")
        for label in pressure, time:
            label.scale_in_place(0.8)
        pressure.next_to(D_axes.y_axis, UP)
        pressure.to_edge(LEFT, buff = MED_SMALL_BUFF)
        time.next_to(D_axes.x_axis.get_right(), DOWN+LEFT)
        axes.labels = VGroup(pressure, time)

        
        self.play(
            FadeIn(pressure),
            ShowCreation(D_axes.y_axis)
        )
        self.play(
            Write(time),
            ShowCreation(D_axes.x_axis)
        )
        self.play(
            GrowFromCenter(D_label),
        )
        self.play(
            ShowCreation(graph, run_time = 4, rate_func=linear),
        )

        self.set_variables_as_attrs(
            D_label,
            D_graph = graph
        )
 

    def get_wave_graph(self, frequency, axes):
        tail_len = 3.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.7*np.cos(2*np.pi*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph


class Instruments(Scene):
    def construct(self):
        self.show_both()
        self.list_instruments()
        self.forced_motion()
        self.wait(5)

    def show_both(self):
        question = TextMobject("Wie produzieren Instrumente Töne?")
        question.scale_in_place(1.3)
        self.violin = SVGMobject("physik/violin.svg")
        sax = SVGMobject("physik/sax.svg", stroke_width=0.1)
        string = TextMobject("Saiteninstrumente")
        self.string_heading = deepcopy(string)
        self.string_heading.shift(3*UP)
        self.string_heading.scale_in_place(1.3)
        winds = TextMobject("Blasinstrumente")
        string.scale_in_place(0.8)
        winds.scale_in_place(0.8)
        self.violin.shift(3*RIGHT+1.5*DOWN)
        self.violin.scale_in_place(1.7)
        sax.scale_in_place(1.5)
        sax.shift(3*LEFT+1.5*DOWN)
        string.shift(3*RIGHT+UP)
        winds.shift(3*LEFT+UP)
        self.play(GrowFromCenter(question))
        self.wait(2)
        headline = deepcopy(question)
        headline.shift(2.5*UP)
        self.play(Transform(question,headline))
        self.wait(1)
        self.play(FadeIn(winds))
        self.play(FadeIn(sax))
        self.play(FadeIn(string))
        self.play(FadeIn(self.violin))
        self.wait(2)
        self.remove(winds)
        self.remove(sax)
        self.play(FadeOut(question))

        self.play(Transform(string, self.string_heading))
    

    
    def list_instruments(self):
        piano = SVGMobject("physik/piano.svg",stroke_width=0.1)
        guitar = SVGMobject("physik/guitar.svg",stroke_width=0.1)
        cello = SVGMobject("physik/cello.svg",stroke_width=0.1)
        guitar.scale_in_place(0.6)
        cello.scale_in_place(1.6)
        cello.shift(UP+5*LEFT)
        piano.shift(1.5*DOWN+LEFT)
        guitar.shift(5*RIGHT+2*UP)
        self.play(FadeIn(cello))
        self.play(FadeIn(guitar))
        self.play(FadeIn(piano))
        self.wait(5)
        self.remove(self.violin,guitar,piano,cello)

    def forced_motion(self):
        cello = SVGMobject("physik/cello.svg",stroke_width=0.1)
        cello.scale_in_place(2.5)
        cello.shift(0.5*DOWN)
        #big_violin.move_to(3*LEFT+1.5*UP)
        #big_violin.scale_in_place(1.5)
        bow = SVGMobject("physik/bow.svg",stroke_width=0.1) 
        bow.scale_in_place(0.1)
        bow.shift(1.1*DOWN)
        bow1 = bow.deepcopy()
        bow2 = bow.deepcopy()
        bow1.shift(RIGHT)
        bow2.shift(LEFT)
        self.play(FadeIn(cello))
        self.wait(1)
        self.play(FadeIn(bow))
        self.play(Transform(bow,bow2),run_time=2)
        self.play(Transform(bow,bow1),run_time=2)
        self.play(Transform(bow,bow2),run_time=2)
        self.play(Transform(bow,bow1),run_time=2)


class StringWithNodes(Scene):
    CONFIG = {
        "frequency" : 0.05,
        "color" : YELLOW,
        "equilibrium_height" : 0,
        "amplitude": 1,
        "plane_kwargs" : {},
    }

    def setup(self):
        
        self.axes = Axes(
            y_min = -2, y_max = 2,
            x_min = -5, x_max = 5,
            number_line_config = {"include_tip" : False},
        )
        # self.headline=TextMobject("Wie produzieren Instrumente Töne?")
        # self.headline.scale_in_place(1.3)
        # self.headline.shift(3*UP)
        left_node= Circle(radius=0.1, fill_color=RED, fill_opacity= 1)
        self.left_node = left_node
        self.left_node.shift(self.axes.x_min*RIGHT)
        right_node= Circle(radius=0.1, fill_color=RED, fill_opacity= 1)
        self.right_node = right_node
        self.right_node.shift(self.axes.x_max*RIGHT)
        center_node= Circle(radius=0.1, fill_color=RED, fill_opacity= 1)
        self.center_node = center_node
        first_third_node= Circle(radius=0.1, fill_color=RED, fill_opacity= 1)
        self.first_third_node = first_third_node
        self.first_third_node.shift(self.axes.x_min/3*RIGHT)
        second_third_node= Circle(radius=0.1, fill_color=RED, fill_opacity= 1)
        self.second_third_node = second_third_node
        self.second_third_node.shift(self.axes.x_max/3*RIGHT)      


    def construct(self):
        #self.play(ShowCreation(self.headline))
        self.pure_frequency()
        self.first_ot()
        self.second_ot()
        self.wave_graph()
        
    
    def pure_frequency(self):
        
        string = self.get_wave_graph()
        #self.string = string
        state1 = self.get_wave_graph() 
        self.amplitude *= -1
        state2 = self.get_wave_graph()        
        #self.state2 = string2
        self.play(FadeIn(self.left_node))
        self.play(FadeIn(self.right_node))
        self.play(ShowCreation(string))
        self.play(Transform(string,state2),run_time=(2))
        self.play(Transform(string,state1),run_time=(2))
        self.play(FadeOut(string))

    def first_ot(self):
        self.amplitude *= -1
        self.frequency = self.frequency*2
        string = self.get_wave_graph()
        self.string = string
        state1 = self.get_wave_graph() 
        self.amplitude *= -1
        state2 = self.get_wave_graph()        
        #self.state2 = string2
        self.play(FadeIn(self.center_node))
        self.play(ShowCreation(string))
        self.play(Transform(string,state2),run_time=(2))
        self.play(Transform(string,state1),run_time=(2))
        self.play(FadeOut(string))
        self.play(FadeOut(self.center_node))



    def second_ot(self):
        self.amplitude *= -1
        self.frequency = self.frequency/2*3
        string = self.get_wave_graph()
        state1 = self.get_wave_graph() 
        self.amplitude *= -1
        state2 = self.get_wave_graph()
        self.play(FadeIn(self.first_third_node))
        self.play(FadeIn(self.second_third_node))               
        self.play(ShowCreation(string))
        self.play(Transform(string,state2),run_time=(2))
        self.play(Transform(string,state1),run_time=(2))
        self.play(FadeOut(string))
        self.play(FadeOut(self.left_node))
        self.play(FadeOut(self.first_third_node))
        self.play(FadeOut(self.second_third_node))
        self.play(FadeOut(self.right_node))
        

    def wave_graph(self):
        graph = self.get_complex_graph()
        self.play(ShowCreation(graph))
        self.wait(5)




    def get_wave_graph(self):
        axes = self.axes
        frequency = self.frequency
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = self.amplitude*np.sin(2*np.pi*frequency*(x-x_min))
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph

    def get_complex_graph(self):
        axes = self.axes
        frequency = 1.2
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = self.amplitude*np.sin(2*np.pi*frequency*(x-x_min))
            value += (self.amplitude/1.5)*np.sin(2*np.pi*frequency*(x-x_min)*2)
            value += (self.amplitude/1)*np.sin(2*np.pi*frequency*(x-x_min)*3)
            value += (self.amplitude/2)*np.sin(2*np.pi*frequency*(x-x_min)*4)
            value += (self.amplitude/1)*np.sin(2*np.pi*frequency*(x-x_min)*5)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph


class ComplexOscillation(Scene):

    CONFIG = {
        "sliding_blocks_config": {
            "block_config": {
                "mass": 1e0,
                "velocity": -2,
            }
        },
        "wait_time": 15,
        "frequency" : 0.5,
        "amplitude":2,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
        "plane_kwargs" : {},
    }
    def setup(self):
        plane = NumberPlane(**self.plane_kwargs)
        self.track_time()
        self.add_points()
       
        

    def add_points(self):
        self.blocks = MovingString(self, **self.sliding_blocks_config)
        self.add(self.blocks)

    def track_time(self):
        time_tracker = ValueTracker()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(time_tracker)
        self.get_time = time_tracker.get_value



    def construct(self):
        self.wait(5)


class AddingFrequencies(Scene):
    CONFIG = {
        "equilibrium_height" : 1.5,
        "amplitude" : 1,
        "freq1" : 2,
        "freq2" : 4,
        "freq3" : 6,
    }
    def setup(self):
        axes = Axes(
                y_min = -2, y_max = 2,
                x_min = 0, x_max = 12,
            )
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = self.amplitude*np.cos(2*np.pi*self.freq1*x) + self.amplitude/2*np.cos(2*np.pi*self.freq2*x) + self.amplitude/3*np.cos(2*np.pi*self.freq3*x)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*self.freq1 + 1
        self.graph = axes.get_graph(func, num_graph_points = int(ngp))
        self.play(FadeIn(self.graph))
    
        
    

    def get_wave_graph(self, frequency, axes):
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.7*np.cos(2*np.pi*frequency*x)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph



class Intro(Scene):
    def construct(self):
        self.show_questions()
        self.wait(5)

    def show_questions(self):
        tone = TextMobject("Was sind Töne?")
        tone.scale_in_place(1.3)
        tone.shift(2*UP)
        instrument = TextMobject("Wie produzieren Instrumente Töne?")
        instrument.scale_in_place(1.3)
        computer = TextMobject("Wie können Computer Instrumente imitieren?")
        computer.scale_in_place(1.3)
        computer.shift(2*DOWN)

        self.play(FadeIn(tone))
        self.wait(0.5)
        self.play(FadeIn(instrument))
        self.wait(1)
        self.play(FadeIn(computer))

    


class HarmonicSeries(Scene):
    CONFIG = {
        "fund_color" : YELLOW,
        "first_color" : PINK,
        "second_color" : TEAL,
        "third_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1,
    }
    def setup(self):
        self.axes = Axes(
            y_min = -2, y_max = 2,
            x_min = -4, x_max = 6,
            number_line_config = {"include_tip" : False},
        )

    
    def construct(self):
        self.fundamental()
        self.first_ot()
        self.second_ot()
        self.third_ot()
        self.combine()
        self.wait(5)

    def fundamental(self):
        axes = self.axes
        axes.shift(1.3*UP)
        frequency = 0.25
        self.graph0 = self.get_wave_graph(frequency,axes)
        self.label0 = TextMobject("Grundton") 
        self.label0.next_to(self.graph0,LEFT)  
        self.play(FadeIn(self.label0))
        self.play(ShowCreation(self.graph0))

    def first_ot(self):
        axes = self.axes
        axes.shift(1.7*DOWN)
        frequency = 0.5
        self.graph1 = self.get_wave_graph(frequency,axes)
        self.label1 = TextMobject("1. Oberton") 
        self.label1.next_to(self.graph1,LEFT)  
        self.play(FadeIn(self.label1))
        self.play(ShowCreation(self.graph1))

    def second_ot(self):
        axes = self.axes
        axes.shift(1.7*DOWN)
        frequency = 0.75
        self.graph2 = self.get_wave_graph(frequency,axes)
        self.label2 = TextMobject("2. Oberton") 
        self.label2.next_to(self.graph2,LEFT)  
        self.play(FadeIn(self.label2))
        self.play(ShowCreation(self.graph2))

    def third_ot(self):
        axes = self.axes
        axes.shift(1.7*DOWN)
        frequency = 1
        self.graph3 = self.get_wave_graph(frequency,axes)
        self.label3 = TextMobject("3. Oberton") 
        self.label3.next_to(self.graph3,LEFT)  
        self.play(FadeIn(self.label3))
        self.play(ShowCreation(self.graph3))

    def combine(self):
        #self.add(self.graph1)
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = -4, x_max = 6,
            number_line_config = {"include_tip" : False},
        )
        axes.shift(1.3*UP)
        def func1(x):
            value =  0.4*np.cos(2*np.pi*0.25*(x-axes.x_min))
            value += 0.4*np.cos(2*np.pi*0.5*(x-axes.x_min))
            return value + self.equilibrium_height
        ngp = 2*(axes.x_max - axes.x_min)*1 + 1
        sum1 = axes.get_graph(func1, num_graph_points = int(ngp))
        def func2(x):
            value =  0.3*np.cos(2*np.pi*0.25*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.5*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.75*(x-axes.x_min))
            return value + self.equilibrium_height
        ngp = 2*(axes.x_max - axes.x_min)*1 + 1
        sum2 = axes.get_graph(func2, num_graph_points = int(ngp))
        def func3(x):
            value =  0.3*np.cos(2*np.pi*0.25*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.5*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.75*(x-axes.x_min))
            value += 0.25*np.cos(2*np.pi*0.1*(x-axes.x_min))
            return value + self.equilibrium_height
        ngp = 2*(axes.x_max - axes.x_min)*1 + 1
        sum3 = axes.get_graph(func3, num_graph_points = int(ngp))

        self.remove(self.label0, self.label1, self.label2, self.label3)
        self.play(ReplacementTransform(self.graph1, self.graph0, run_time=0.25))
        self.play(ReplacementTransform(self.graph0, sum1, run_time=0.4))
        self.play(ReplacementTransform(self.graph2, sum1, run_time=0.6))
        self.play(ReplacementTransform(sum1, sum2,run_time=0.3))
        self.play(ReplacementTransform(self.graph3, sum2, run_time=0.8))
        self.play(ReplacementTransform(sum2, sum3,run_time=0.3))
        axes.x_min = -8
        axes.x_max = 8
        def func4(x):
            value =  0.3*np.cos(2*np.pi*0.25*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.5*(x-axes.x_min))
            value += 0.3*np.cos(2*np.pi*0.75*(x-axes.x_min))
            value += 0.25*np.cos(2*np.pi*0.1*(x-axes.x_min))
            return value + self.equilibrium_height
        ngp = 2*(axes.x_max - axes.x_min)*1 + 1
        centered_sum = axes.get_graph(func3, num_graph_points = int(ngp))
        centered_sum.shift(2.5*DOWN)
        self.play(Transform(sum3,centered_sum))


    

    


    def get_wave_graph(self, frequency, axes):
        tail_len = 0.5
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.5*np.cos(2*np.pi*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph



class Fourier(GraphScene):
    CONFIG={
        "axes_config" : {
            "x_min": 0,
            "x_max" : 12,
            "y_min" : -6,
            "y_max": 6,
            "y_axis_config": {
                "unit_size" : 0.15,
                "tick_frequency" : 16,
            },
        },
        "frequency_axes_config" : {
            "x_min":0,
            "x_max":2,
            "x_axis_config" : {
                "unit_size" : 6,
                "tick_frequency" : 0.25,
                },
            "y_min":0,
            "y_max":1,
            "y_axis_config" : {
                "unit_size" : 2, 
                },
            "x_labeled_nums" :range(0,2,1),
        },
        "equilibrium_height": 1,
    }

    def break_apart_and_add(self):
        axes = Axes(**self.axes_config)
        axes.to_corner(UP+LEFT)
        func = lambda t : sum([
            np.cos(TAU*f*t)
            for f in (0.5, 1, 0.75, 1.5)
        ])
        summed_graph = axes.get_graph(func)
        summed_graph.set_color(RED)
        sum_label = TextMobject ("f(x) = sin(1x)+sin(1.5x)+sin(2x)+sin(3x)")
        sum_label.next_to(summed_graph,DOWN)
        fund = self.get_wave_graph(0.5,axes)
        fund.to_corner(DOWN,LEFT)
        fund.shift(UP)
        first_ot = self.get_wave_graph(0.75,axes)
        first_ot.next_to(fund,3*UP)
        second_ot = self.get_wave_graph(1,axes)
        second_ot.next_to(first_ot,3*UP)
        third_ot = self.get_wave_graph(1.5,axes)
        third_ot.next_to(second_ot,3*UP)
        self.play(FadeIn(axes))
        self.play(ShowCreation(summed_graph))
        self.wait(5)
        self.play(TransformFromCopy(summed_graph, fund))
        self.play(TransformFromCopy(summed_graph, first_ot))
        self.play(TransformFromCopy(summed_graph, second_ot))
        self.play(TransformFromCopy(summed_graph, third_ot))
        self.wait(5)
        first_ot_copy = first_ot.copy()
        first_ot_copy.move_to(fund)
        self.play(ReplacementTransform(first_ot,first_ot_copy))
        second_ot_copy = second_ot.copy()
        second_ot_copy.move_to(fund)
        self.play(ReplacementTransform(second_ot,second_ot_copy))
        third_ot_copy = third_ot.copy()
        third_ot_copy.move_to(fund)
        self.play(ReplacementTransform(third_ot,third_ot_copy))
        func_copy = lambda t : sum([
            2*np.cos(TAU*f*t)
            for f in (0.5, 1, 0.75, 1.5)
        ])
        summed_copy = axes.get_graph(func)
        summed_copy.move_to(fund)
        summed_copy.set_color(BLUE)
        summed_copy_2 = summed_copy.copy()
        summed_copy_2.to_corner(UP + LEFT)
        summed_copy_2.shift(3*DOWN)
        self.play(
            ShowCreation(summed_copy),
            FadeOut(fund),
            FadeOut(first_ot_copy),
            FadeOut(second_ot_copy),
            FadeOut(third_ot_copy)
            )
        self.play(
            FadeOut(axes),
            FadeOut(summed_graph),
            ReplacementTransform(summed_copy, summed_copy_2)
            )
        self.sum = summed_copy_2       
        self.axes = axes
        # axes.to_center()
        # graph_1 = self.get_wave_graph(axes)
        # self.play(Transform(self.sum, graph1))


    def small_changes(self):
        #summ = self.sum   
        axes = Axes(**self.axes_config)
        axes.to_corner(UP + LEFT)
        axes.shift(3*DOWN)
        graph_1 = self.get_complex_graph(axes, 3, 1, 3)
        graph_1.set_color(GREEN)
        graph_2 = self.get_complex_graph(axes, 5, 1, 1)
        graph_2.set_color(YELLOW)
        graph_3 = self.get_complex_graph(axes, 1, 2, 2)
        graph_3.set_color(RED)
        graph_4 = self.get_complex_graph(axes, 4, 1.5, 1)
        graph_4.set_color(BLUE)
        self.play(ReplacementTransform(self.sum, graph_1, run_time = 1.5))
        self.play(ReplacementTransform(graph_1, graph_2, run_time = 1.5))
        self.play(ReplacementTransform(graph_2, graph_3, run_time = 1.5))
        self.play(ReplacementTransform(graph_3, graph_4, run_time = 1.5))


        
    def get_wave_graph(self, frequency, axes):
        tail_len = 0.5
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.5*np.cos(TAU*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph

    def get_complex_graph(self, axes, tail_len, amp1, amp2):
        #tail_len = 0.5
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = np.sin(2*np.pi*0.5*x) + amp1*np.sin(2*np.pi*0.75*x) + amp2*np.sin(2*np.pi*1*x) + np.sin(2*np.pi*1.5*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*1.5 + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph
        


    


    def construct(self):
        self.break_apart_and_add()
        self.wait(3)
        self.small_changes()
        self.wait(5)


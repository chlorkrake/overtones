# -*- coding: utf-8 -*-
from constants import *
import scipy.integrate

from manimlib.imports import *


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
        "radius": 0.1,
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
        "amplitude": 0.2,
        "frequency":5,
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
            for x in np.arange(-2,6,0.25)
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
        #point_field = self.point_field
        self.timer +=dt
        ps_block= self.amplitude* np.sin(self.frequency* self.timer)
        block.move_to(
                (self.membrane+ ps_block) * RIGHT +
                floor_y * UP,      
                DL,
            )
            
        for point in self.point_field:
            ps_point = point.x + self.amplitude*np.cos(self.frequency * (self.timer - point.x))
            point.move_to(
                (point.x+ ps_point) * RIGHT +
                point.y * UP,      
                DL,
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
        
 
class AirPressure(Scene):
    CONFIG = {
        "sliding_blocks_config": {
            "block_config": {
                "mass": 1e0,
                "velocity": -2,
            }
        },
        "wait_time": 15,
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
        self.wait(15)
        


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
        self.wait(5)

    def show_both(self):
        question = TextMobject("Wie produzieren Instrumente Töne?")
        question.scale_in_place(1.3)
        violin = SVGMobject("physik/violin.svg")
        sax = SVGMobject("physik/sax.svg")
        string = TextMobject("Saiteninstrumente")
        self.string_heading = deepcopy(string)
        self.string_heading.shift(3*UP)
        self.string_heading.scale_in_place(1.3)
        winds = TextMobject("Blasinstrumente")
        string.scale_in_place(0.8)
        winds.scale_in_place(0.8)
        violin.shift(3*RIGHT+1.5*DOWN)
        violin.scale_in_place(1.7)
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
        self.wait(1)
        self.play(FadeIn(string))
        self.play(FadeIn(violin))
        self.wait(2)
        self.play(FadeOut(winds))
        self.play(FadeOut(sax))
        self.play(FadeOut(question))

        self.play(Transform(string, self.string_heading))
    

    
    def list_instruments(self):
        piano = SVGMobject("physik/piano.svg")
        guitar = SVGMobject("physik/guitar.svg")
        cello = SVGMobject("physik/cello.svg")
        guitar.scale_in_place(0.6)
        cello.scale_in_place(1.6)
        cello.shift(UP+5*LEFT)
        piano.shift(1.5*DOWN+LEFT)
        guitar.shift(5*RIGHT+2*UP)
        self.play(FadeIn(cello))
        self.play(FadeIn(guitar))
        self.play(FadeIn(piano))


        
    


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
        self.headline=TextMobject("Wie produzieren Instrumente Töne?")
        self.headline.scale_in_place(1.3)
        self.headline.shift(3*UP)
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
        self.play(ShowCreation(self.headline))
        self.pure_frequency()
        self.first_ot()
        self.second_ot()
        
    
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

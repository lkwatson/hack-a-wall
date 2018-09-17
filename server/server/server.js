//npm modules
const express = require('express');
const uuid = require('uuid/v4')
const session = require('express-session')
const FileStore = require('session-file-store')(session);
const bodyParser = require('body-parser');
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const axios = require('axios');
const bcrypt = require('bcrypt-nodejs');

// configure passport.js to use the local strategy
passport.use(new LocalStrategy(
  { usernameField: 'email' },
  (email, password, done) => {
    axios.get(`http://localhost:5000/users?email=${email}`)
    .then(res => {
      const user = res.data[0]
      if (!user) {
        return done(null, false, { message: 'Invalid credentials.\n' });
      }
      if (!bcrypt.compareSync(password, user.password)) {
        return done(null, false, { message: 'Invalid credentials.\n' });
      }
      return done(null, user);
    })
    .catch(error => done(error));
  }
));

// tell passport how to serialize the user
passport.serializeUser((user, done) => {
  done(null, user.id);
});

passport.deserializeUser((id, done) => {
  axios.get(`http://localhost:5000/users/${id}`)
  .then(res => done(null, res.data) )
  .catch(error => done(error, false))
});

// create the server
const app = express();
var server = require('http').createServer(app);  
var io = require('socket.io').listen(server);

// add & configure middleware
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())
app.use(session({
  genid: (req) => {
    return uuid() // use UUIDs for session IDs
  },
  store: new FileStore(),
  secret: 'keyboard cat',
  resave: false,
  saveUninitialized: true
}))
app.use(passport.initialize());
app.use(passport.session());

// create the homepage route at '/'
app.use(express.static(__dirname + '/node_modules'));  
app.use('/images', express.static(__dirname + '/images'));
app.get('/', function(req, res,next) {  
    res.sendFile(__dirname + '/index.html');
});

// create the login get and post routes
app.get('/login', (req, res) => {
  res.send(`You got the login page!\n`)
})

app.post('/login', (req, res, next) => {
  passport.authenticate('local', (err, user, info) => {
    if (info) {return res.send(info.message)}
    if (err) { return next(err); }
    if (!user) { return res.redirect('/login'); }
    req.login(user, (err) => {
      if (err) { return next(err); }
      return res.redirect('/authrequired');
    })
  })(req, res, next);
})

app.get('/authrequired', (req, res) => {
  if(req.isAuthenticated()) {
    res.send('you hit the authentication endpoint\n')
  } else {
    res.redirect('/')
  }
})

// tell the server what port to listen on
server.listen(3000, () => {
  console.log('Listening on localhost:3000')
})

avatarIndex = 0;
avatars = ["https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/247/bird-512.png",
"https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/241/pig-512.png",
"https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/246/dog-512.png",
"https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/248/fish-512.png",
"https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/247/cow-512.png",
"https://cdn0.iconfinder.com/data/icons/cute-animal-avatar-1/244/sheep-512.png"]

response = {}


io.on('connection', function(client) {
    console.log('Client connected, id=' + client.id);

    client.on('join', function(data) {
        console.log(data);
    });

    client.on('messages', function(data) {
		var jsonArr = JSON.parse(data);
		for (i = 0; i < jsonArr.length; i++) {
			var person = jsonArr[i];
			if (!response.hasOwnProperty(person.person_id)) {
				response[person.person_id] = {};
				response[person.person_id]["avatar"] = avatars[avatarIndex];
				avatarIndex = (avatarIndex + 1) % avatars.length;
			}
			response[person.person_id]["x"] = person["x"];
			response[person.person_id]["y"] = person["y"];
			
			if (typeof person["hand1_x"] != "undefined") {
				response[person.person_id]["hand1_x"] = person["hand1_x"];
			}
			if (typeof person["hand1_y"] != "undefined") {
				response[person.person_id]["hand1_y"] = person["hand1_y"];
			}
			if (typeof person["hand2_x"] != "undefined") {
				response[person.person_id]["hand2_x"] = person["hand2_x"];
			}
			if (typeof person["hand2_y"] != "undefined") {
				response[person.person_id]["hand2_y"] = person["hand2_y"];
			}
			
		}
		client.emit('broad', response);
		client.broadcast.emit('broad', response);
		console.log(response);
    });

});